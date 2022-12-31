import configs
from model_utils import vgg16_model
from data_utils import CustomImageFolder
import torch, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import pil_loader
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from new_dissection.netdissect import nethook, renormalize, imgviz, show
from new_dissection.netdissect.easydict import EasyDict
from new_dissection.experiment import dissect_experiment as experiment



def load_model (model_file=None):
    if configs.model_name == 'vgg16':
        model = vgg16_model(num_classes=configs.num_classes)
    else:
        model = models.__dict__[configs.model_name](num_classes=configs.num_classes)
    model.load_state_dict(torch.load(model_file))
    model = nethook.InstrumentedModel(model).cuda().eval()

    return model



def load_data (dataset_dir=None): 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(configs.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.norm_mean, std=configs.norm_std)
    ])

    dataset = CustomImageFolder(root=dataset_dir, transform=transform)
    #dataset = parallelfolder.ParallelImageFolders([dataset_dir], classification=True, shuffle=True, transform=transform)

    data_loader = DataLoader(dataset, batch_size=configs.batch_size)
    print('Processing {} data examples in {} batches with class index {}'.format(len(dataset), len(data_loader), dataset.class_to_idx))
    
    return dataset, data_loader



def load_channels_data (tally_path):
    f = open(tally_path)
    tally_data = json.load(f)
    if "units" not in tally_data:
        print('Error: units not present in loaded tally data from path {}'.format(tally_path))
        return None

    f.close()
    channels_data = tally_data["units"]
    channels_map = {}

    for ch_item in channels_data:
        if (('iou' not in ch_item) or ('unit' not in ch_item) or ('label' not in ch_item) or 
            ('cat' not in ch_item) or ('low_thresh' not in ch_item) or ('high_thresh' not in ch_item)): 
            print('Error: incomplete data in channel item:', ch_item)
            continue

        channel = ch_item['unit'] + 1
        channels_map[channel] = {
            'concept': ch_item['label'],
            'category': ch_item['cat'],
            'low_thresh': ch_item['low_thresh'],
            'high_thresh': ch_item['high_thresh'],
            'is_valid': True if ch_item['iou'] > configs.min_iou else False
        }

    channels = list(set([k for k,v in channels_map.items()]))
    concepts = list(set([v['concept'] for k,v in channels_map.items()]))

    channels.sort()
    concepts.sort()

    print('Processing {} concepts and {} channels'.format(len(concepts), len(channels)))
    print('channels_map:', channels_map)

    return channels_map, channels, concepts



def get_binned_predictions (logits):
    # If the softmax probability for any of the other classes is higher than the certainty_ratio (e.g. 2/3) of the top class probability, 
    # then it will be considered as `maybe` instead of certain. 
    # Number for maybe status of each class is equal to the class number plus total number of classes (e.g. 2 for 0 class in a binary setting).
    certainty_ratio = (1.0 - configs.certainty_thresh) / configs.certainty_thresh

    probs = torch.softmax(logits, dim=1)
    top_probs, preds = torch.max(probs, dim=1)
    top_probs = (top_probs * certainty_ratio).view(-1, 1).expand(-1, probs.shape[1])
    probs_mask = probs > top_probs
    probs_mask_sum = torch.sum(probs_mask, dim=1)
    
    for i in range(preds.shape[0]):
        s = probs_mask_sum[i]
        if s > 1: 
            preds[i] = preds[i] + configs.num_classes

    return preds.cpu().numpy()



def plot_activation_histogram (act):
    hist, bin_edges = torch.histogram(act, bins=10)
    hist = hist.tolist()
    bin_edges = bin_edges.tolist()
    print(hist)
    print(bin_edges)
    x = []
    for i in range(len(bin_edges)-1):
        x.append('{:.4f}-{:.4f}'.format(bin_edges[i], bin_edges[i+1]))

    plt.figure(figsize=(15, 15))
    plt.bar(x, hist, align="center")
    plt.xlabel('Activation/Gradient Value')
    plt.ylabel('Frequency')
    plt.show()



def plot_sample_image_activations (dataset, image, channel, seg, act, grad, act_grad, upact_grad, 
                                    channels_map, seg_concept_index_map, image_threshs):
    act = act[None, :, :, :]
    grad = grad[None, :, :, :]
    act_grad = act_grad[None, :, :, :]
    iv = imgviz.ImageVisualizer(size=(configs.image_size, configs.image_size), 
                                image_size=(configs.image_size, configs.image_size), source=dataset)

    img = renormalize.as_image(image, source=dataset)
    ch_index = channel - 1
    ch_info = channels_map[channel]
    concept = ch_info['concept']
    category = ch_info['category']
    act_thresh = ch_info['high_thresh']
    grad_thresh = image_threshs['high_thresh']
    print('Visualizing filter {} mapped to concept {} from category {}, with activation thresh {} and gradient thresh {}'
        .format(channel, concept, category, act_thresh, grad_thresh))

    act_grad_high_mask = (upact_grad[ch_index] > grad_thresh)

    seg_concept_index = seg_concept_index_map[concept]
    cat_index = configs.category_index_map[category]
    target_seg = seg[cat_index]
    target_seg_concept_mask = (target_seg == seg_concept_index)

    overlap_mask = act_grad_high_mask & target_seg_concept_mask

    print('Input image:')
    show([[img]])

    print('Segmentation mask:')
    show([[iv.segmentation(target_seg_concept_mask)]])   # seg[0]

    print('Activation-gradients mask:')
    show([[iv.segmentation(act_grad_high_mask)]])

    print('Segmentation and activation-gradient overlap mask:')
    show([[iv.segmentation(overlap_mask)]])

    print('Activations heatmap:')
    show([[iv.heatmap(act, (0, ch_index), mode='nearest')]])

    print('Gradients heatmap:')
    print(grad[0, ch_index])
    print(grad[0, ch_index].min(), grad[0, ch_index].max())
    show([[iv.heatmap(grad, (0, ch_index), mode='nearest')]])

    print('Activation-gradients heatmap:')
    show([[iv.heatmap(act_grad, (0, ch_index), mode='nearest')]])

    print('Segmentation highlighted:')
    show([[iv.masked_image(image, target_seg_concept_mask.float(), level=0.99)]])

    print('Activations highlighted:')
    show([[iv.masked_image(image, act, (0, ch_index), level=act_thresh)]])

    print('Activation-gradients highlighted:')
    show([[iv.masked_image(image, act_grad, (0, ch_index), level=grad_thresh)]])

    print('Segmentation and activation-gradient overlap highlighted:')
    show([[iv.masked_image(image, overlap_mask.float(), level=0.99)]])



def extract_concepts_from_image (act, upact, seg, path, channels_map, seg_concept_index_map, channels, concepts, image_threshs=None):
    num_channels = act.shape[0]
    image_concepts = {con:configs.low_value for con in concepts}  # used to hold the high/mid/low value of each concept for this image (used for image concepts file and saving activation images process)
    image_channels = {ch:configs.low_value for ch in channels}    # used to hold the high/mid/low value of each channel for this image (used for image channels file and saving activation images process)
    image_concepts_counts = {con:0 for con in concepts}   # used to count how many channels related to each concept were high for this image (just used for stat keeping purposes)
    image_channels_counts = {ch:0 for ch in channels}     # used to keep the number of mid/high thresh pixels for each channel for this image (used in saving activation images process)

    for ch in channels:
        ch_index = ch - 1
        ch_upact = upact[ch_index]
        ch_info = channels_map[ch]
        is_valid = ch_info['is_valid']
        channel_concept = ch_info['concept']
        channel_category = ch_info['category']
        channel_high_thresh = image_threshs['high_thresh'] if configs.check_gradients else ch_info['high_thresh']
        channel_low_thresh = image_threshs['low_thresh'] if configs.check_gradients else ch_info['low_thresh']

        if (ch_upact is None) or (not is_valid) or (channel_concept is None) or (channel_category is None) or (channel_high_thresh is None):
            continue

        # Checking whether the channel concept can be considered as high value for this image: 
        is_high = False
        ch_upact_high_mask = (ch_upact > channel_high_thresh)
        num_high_thresh = np.sum(ch_upact_high_mask.numpy())

        if num_high_thresh >= configs.min_thresh_pixels:
            if configs.check_seg_overlap:
                seg_concept_index = seg_concept_index_map[channel_concept] if channel_concept in seg_concept_index_map else None
                cat_index = configs.category_index_map[channel_category] if channel_category in configs.category_index_map else None
                if (seg_concept_index is None) or (cat_index is None):
                    print('Error: Missing segmentation concept index ({}) or category index ({}) for channel {} mapped to concept {} from category {}'
                        .format(seg_concept_index, cat_index, ch, channel_concept, channel_category))
                else:
                    target_seg = seg[cat_index]
                    target_seg_concept_mask = (target_seg == seg_concept_index)
                    num_concept_seg = np.sum(target_seg_concept_mask.numpy())
                    overlap_mask = ch_upact_high_mask & target_seg_concept_mask
                    num_overlap = np.sum(overlap_mask.numpy())

                    if configs.overlap_mode == 'overlap_to_union_ratio':
                        union_mask = ch_upact_high_mask | target_seg_concept_mask
                        num_union = np.sum(union_mask.numpy())
                        overlap_ratio = num_overlap / num_union
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_high = True

                    elif configs.overlap_mode == 'overlap_to_activation_ratio':
                        overlap_ratio = num_overlap / num_high_thresh
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_high = True

                    elif configs.overlap_mode == 'overlap_to_segmentation_ratio':
                        overlap_ratio = num_overlap / num_concept_seg
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_high = True

                    else:
                        if num_overlap >= configs.min_overlap_pixels:
                            is_high = True

            else:
                is_high = True

        if is_high:
            image_channels[ch] = configs.high_value
            image_concepts[channel_concept] = configs.high_value
            image_concepts_counts[channel_concept] += 1
            image_channels_counts[ch] = num_high_thresh
            continue
        
        if not configs.binning_features:
            continue

        # Checking whether the channel concept can be considered as mid value for this image: 
        is_mid = False
        ch_upact_mid_mask = (ch_upact > channel_low_thresh)
        num_mid_thresh = np.sum(ch_upact_mid_mask.numpy())

        if num_mid_thresh >= configs.min_thresh_pixels:
            if configs.check_seg_overlap:
                seg_concept_index = seg_concept_index_map[channel_concept] if channel_concept in seg_concept_index_map else None
                cat_index = configs.category_index_map[channel_category] if channel_category in configs.category_index_map else None
                if (seg_concept_index is None) or (cat_index is None):
                    print('Error: Missing segmentation concept index ({}) or category index ({}) for channel {} mapped to concept {} from category {}'
                        .format(seg_concept_index, cat_index, ch, channel_concept, channel_category))
                else:
                    target_seg = seg[cat_index]
                    target_seg_concept_mask = (target_seg == seg_concept_index)
                    num_concept_seg = np.sum(target_seg_concept_mask.numpy())
                    overlap_mask = ch_upact_mid_mask & target_seg_concept_mask
                    num_overlap = np.sum(overlap_mask.numpy())

                    if configs.overlap_mode == 'overlap_to_union_ratio':
                        union_mask = ch_upact_mid_mask | target_seg_concept_mask
                        num_union = np.sum(union_mask.numpy())
                        overlap_ratio = num_overlap / num_union
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_mid = True

                    elif configs.overlap_mode == 'overlap_to_activation_ratio':
                        overlap_ratio = num_overlap / num_mid_thresh
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_mid = True

                    elif configs.overlap_mode == 'overlap_to_segmentation_ratio':
                        overlap_ratio = num_overlap / num_concept_seg
                        if overlap_ratio >= configs.min_overlap_ratio:
                            is_mid = True

                    else:
                        if num_overlap >= configs.min_overlap_pixels:
                            is_mid = True

            else:
                is_mid = True

        if is_mid:
            image_channels[ch] = configs.mid_value
            image_channels_counts[ch] = num_mid_thresh
            if image_concepts[channel_concept] != configs.high_value:
                image_concepts[channel_concept] = configs.mid_value

    return image_concepts, image_channels, image_concepts_counts, image_channels_counts



def extract_concepts (model, segmodel, upfn, renorm, data_loader, channels_map, seg_concept_index_map, channels, concepts):
    activations = []
    gradients = []

    def activations_hook (module, input, output):
        activations.append(output.detach().cpu())
        if configs.check_gradients:
            output.register_hook(gradients_hook)

    def gradients_hook (grad):
        gradients.append(grad.detach().cpu())

    layer = model._modules.get(configs.target_layer)
    if layer is None:
        for n,l in model.named_modules():
            if n == 'model.' + configs.target_layer:
                layer = l
                print('Target layer found:', n)
                break
    layer.register_forward_hook(activations_hook)

    model.eval()

    concepts_counts = {con:0 for con in concepts}
    channels_counts = {ch:0 for ch in channels}
    concepts_counts_by_class = {c:{con:0 for con in concepts} for c in configs.classes}

    concepts_rows_list = []
    channels_rows_list = []
    image_channels_counts_list = []
    image_threshs_list = []
    acts_list = []
    num_images = 0
    total_acc = 0
    
    for i, (images, labels, paths) in tqdm(enumerate(data_loader)):
        del activations[:]
        del gradients[:]
        images_gpu = images.cuda()
        labels_gpu = labels.cuda()

        model.zero_grad()
    
        output = model(images_gpu)
        _, preds = torch.max(output, 1)

        acts = activations[0]
        raw_acts = acts

        grads = None
        if configs.check_gradients:
            one_hot = torch.zeros_like(output).cuda()
            one_hot.scatter_(1, preds[:, None], 1.0)
            output.backward(gradient=one_hot, retain_graph=True)
            grads = gradients[0]

            wgrads = grads
            if configs.pool_gradients:
                wgrads = torch.mean(grads, dim=[2,3], keepdims=True)

            acts = acts * wgrads
            acts = F.relu(acts)
            # if i == 0:
            #     print('output: {}, preds: {}, one_hot: {}, grads: {}, wgrads: {}'
            #         .format(output.shape, preds.shape, one_hot.shape, grads.shape, wgrads.shape))

        total_acc += (preds == labels_gpu).float().sum()
        preds = preds.cpu().numpy()
        labels = labels.numpy()

        if configs.binning_classes:
            preds = get_binned_predictions(output)

        upacts = upfn(acts)
        segs = None
        if configs.check_seg_overlap:
            segs = segmodel.segment_batch(renorm(images_gpu), downsample=4).cpu()

        # if i == 0:
        #     print('images: {}, labels: {}, preds: {}'.format(images.shape, labels.shape, preds.shape))
        #     print('acts: {}, upacts: {}, grads: {}, segs: {}'
        #         .format(acts.shape, upacts.shape, grads.shape if grads != None else 0, segs.shape if segs != None else 0))

        for j in range(images.shape[0]):
            num_images += 1
            pred = preds[j]
            label = labels[j]
            path = paths[j]
            image = images[j]
            act = acts[j]
            upact = upacts[j]
            raw_act = raw_acts[j]
            seg = segs[j] if segs != None else None
            grad = grads[j] if grads != None else None
            _, fname = os.path.split(path)

            # In case of checking gradients, we compute the specific activation/gradient threshold 
            # for an image based on the range of the values of all the channels of the image: 
            image_threshs = {'high_thresh': 0, 'low_thresh': 0}
            if configs.check_gradients:
                image_threshs['high_thresh'] = torch.quantile(act, q=configs.gradient_high_thresh).item()
                if configs.binning_features:
                    image_threshs['low_thresh'] = torch.quantile(act, q=configs.gradient_low_thresh).item()

            image_concepts, image_channels, image_concepts_counts, image_channels_counts = \
                extract_concepts_from_image(act, upact, seg, path, channels_map, seg_concept_index_map, channels, concepts, image_threshs)

            # if (i == 0) and (j == 0):
            #     print('Image {} with label {}, pred {}, act {}, upact {}, high threshold {}, low threshold {}, and concepts {}'
            #         .format(path, label, pred, act.shape, upact.shape, image_threshs['high_thresh'], image_threshs['low_thresh'], image_concepts))
            #     print('Image {} with min activation {}, max activation {}, and thresholds {}'.format(fname, act.min(), act.max(), image_threshs))
            #     plot_activation_histogram(act)

            # Test visualizations: 
            # if configs.check_seg_overlap and configs.check_gradients and (fname == '00000133.jpg'):
            #     channel = 170
            #     plot_sample_image_activations(data_loader.dataset, image, channel, seg, raw_act, grad, act, upact, channels_map, seg_concept_index_map, image_threshs)
            #     return

            # if configs.check_seg_overlap and configs.check_gradients and ('sea' in image_concepts) and (image_concepts['sea'] == configs.high_value):
            #     channel = 144
            #     print('Image {} with pred {} and label {}'.format(fname, pred, label))
            #     plot_sample_image_activations(data_loader.dataset, image, channel, seg, raw_act, grad, act, upact, channels_map, seg_concept_index_map, image_threshs)

            acts_list.append(act)
            image_channels_counts_list.append(image_channels_counts)
            image_threshs_list.append(image_threshs)

            image_concepts_row = image_concepts
            image_concepts_row['pred'] = pred
            image_concepts_row['label'] = label
            image_concepts_row['id'] = num_images
            image_concepts_row['file'] = fname
            image_concepts_row['path'] = path
            concepts_rows_list.append(image_concepts_row)

            image_channels_row = image_channels
            image_channels_row['pred'] = pred
            image_channels_row['label'] = label
            image_channels_row['id'] = num_images
            image_channels_row['file'] = fname
            image_channels_row['path'] = path
            channels_rows_list.append(image_channels_row)
            
            for con in concepts:
                cnt = image_concepts_counts[con]
                val = 1 if cnt > 0 else 0
                concepts_counts[con] += val
                concepts_counts_by_class[pred][con] += val

            for ch in channels:
                cnt = image_channels_counts[ch]
                channels_counts[ch] += 1 if cnt > 0 else 0

    total_acc = total_acc / num_images
    print('\nExtracted concepts from {} images with accuracy {:.3f}.'.format(num_images, total_acc))
    print('\nConcept counts:', concepts_counts)
    for c,counts in concepts_counts_by_class.items():
        print('\nConcept counts of class {}: {}'.format(c, counts))
    print('\nChannel counts:', channels_counts)

    concepts_df = pd.DataFrame(concepts_rows_list)
    channels_df = pd.DataFrame(channels_rows_list)

    return concepts_df, channels_df, acts_list, image_channels_counts_list, image_threshs_list



def filter_extracted_concepts (concepts_df, channels_df, channels_map):
    preds_df = concepts_df['pred']
    meta_cols = ['pred', 'label', 'id', 'file', 'path']
    meta_df = concepts_df[meta_cols]
    concept_cols = list(set(concepts_df.columns) - set(meta_cols))
    concept_cols.sort()
    cons_df = concepts_df[concept_cols]

    initial_concepts = list(cons_df.columns)
    print('Initial concepts ({}): {}'.format(len(initial_concepts), initial_concepts))
    
    var_selector = VarianceThreshold(threshold=(configs.low_variance_thresh * (1 - configs.low_variance_thresh)))
    var_selector.fit(cons_df)
    var_col_indices = var_selector.get_support(indices=True)
    cons_df = cons_df.iloc[:,var_col_indices]
    var_filtered_concepts = list(cons_df.columns)
    var_removed_concepts = set(initial_concepts) - set(var_filtered_concepts)
    print('Concepts removed by variance filtering ({}): {}'.format(len(var_removed_concepts), var_removed_concepts))

    k = configs.max_concepts if len(var_filtered_concepts) > configs.max_concepts else 'all'
    mut_selector = SelectKBest(mutual_info_classif, k=k)
    mut_selector.fit(cons_df, preds_df)
    mut_col_indices = mut_selector.get_support(indices=True)
    cons_df = cons_df.iloc[:,mut_col_indices]
    filtered_concepts = list(cons_df.columns)
    mut_removed_concepts = set(var_filtered_concepts) - set(filtered_concepts)
    print('Concepts removed by mutual info filtering ({}): {}'.format(len(mut_removed_concepts), mut_removed_concepts))
    print('Concepts reduced from {} to {} by concept filtering.'.format(len(initial_concepts), len(filtered_concepts)))
    print('Final concepts after filtering ({}): {}'.format(len(filtered_concepts), filtered_concepts))

    filtered_concepts_df = pd.concat([cons_df, meta_df], axis=1)
    # display(filtered_concepts_df.head())

    channel_cols = list(set(channels_df.columns) - set(meta_cols))
    filtered_channels = [ch for ch in channel_cols if (channels_map[ch]['concept'] in filtered_concepts)]
    print('Channels reduced from {} to {} by concept filtering.'.format(len(channel_cols), len(filtered_channels)))

    cols_to_keep = filtered_channels + meta_cols
    filtered_channels_df = channels_df[cols_to_keep]
    # display(filtered_channels_df.head())

    return filtered_concepts_df, filtered_channels_df, filtered_concepts, filtered_channels



def save_activation_images_of_image (iv, image_index, image_path, image_fname, acts, img_concepts_row, img_channels_row, 
                                     image_channels_counts, channels_map, concepts, output_dir, image_threshs=None):
    acts = acts[None, :, :, :]   # as required by iv.masked_image
    # if image_index == 1:
    #     print('acts.shape in save_activation_images:', acts.shape)

    image_activated_channels = [k for k,v in image_channels_counts.items() if v > 0]   # Only keep those channels which have been either mid or high for the image
    if len(image_activated_channels) == 0:
        # print('Image {} with path {} has no activated channels!'.format(image_index, image_path))
        return 0

    image_concept_channels = {con:[] for con in concepts}
    for ch in image_activated_channels:
        ch_info = channels_map[ch]
        is_valid = ch_info['is_valid']
        channel_concept = ch_info['concept']
        con_value = img_concepts_row[channel_concept]
        ch_value = img_channels_row[ch]
        num_high_thresh = image_channels_counts[ch]   # In case of binning features, it can be the count of either mid or high pixels, depending on whether the channel has been mid or high for the image

        if not is_valid:   # In case the channel IoU with the concept is lower than the min threshold, we don't need an image saved for the channel
            continue

        if con_value != ch_value:   # In case the concept is high for the image and the channel is mid, we don't need an image saved for the channel
            continue
            
        image_concept_channels[channel_concept].append((ch, num_high_thresh))

    image = pil_loader(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(configs.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.norm_mean, std=configs.norm_std)
    ])
    image = transform(image)

    images = []
    filenames = []
    for con,lst in image_concept_channels.items():
        if len(lst) == 0:
            continue

        top_channel_nums = sorted(lst, key=lambda x: x[1], reverse=True)[:configs.n_top_channels_per_concept]
        top_channels = [k for k,v in top_channel_nums]
        # if image_index == 1:
        #     print('Top channels for concept {}: {}'.format(con, top_channel_nums))

        for i,ch in enumerate(top_channels):
            ch_index = ch - 1
            ch_info = channels_map[ch]
            ch_value = img_channels_row[ch]

            channel_thresh = image_threshs['high_thresh'] if configs.check_gradients else ch_info['high_thresh']
            feature_value_title = ''

            if configs.binning_features:
                if ch_value == configs.high_value:
                    feature_value_title = 'high_'
                else:
                    feature_value_title = 'mid_'
                    channel_thresh = image_threshs['low_thresh'] if configs.check_gradients else ch_info['low_thresh']

            if (channel_thresh is None):
                print('Error: Missing activation or threshold ({}) for channel {}!'.format(channel_thresh, ch))
                continue

            new_image = iv.masked_image(image, acts, (0, ch_index), level=channel_thresh)

            ind = image_fname.rfind('.')
            image_fname_raw = image_fname[:ind]
            new_fname = image_fname_raw + '_' + feature_value_title + str(ch) + '_' + con + '.jpg'
            new_path = os.path.join(output_dir, new_fname)

            images.append(new_image)
            filenames.append(new_path)

            new_image.save(new_path, optimize=True, quality=99)

    # print('Saved {} activation images for image {} with path {}'.format(len(images), image_index, image_path))
    return len(images)



def save_image_concepts_dataset (concepts_df, channels_df, image_channels_counts_list, image_threshs_list, acts_list, upfn, dataset, channels_map, 
                                 filtered_concepts, filtered_channels, concepts_output_path, channels_output_path, activation_images_path):
    iv = imgviz.ImageVisualizer(size=(configs.image_size, configs.image_size), 
                                image_size=(configs.image_size, configs.image_size), source=dataset)

    num_images = len(concepts_df.index)

    filtered_concepts_counts = {con:0 for con in filtered_concepts}
    filtered_channels_counts = {ch:0 for ch in filtered_channels}
    filtered_concepts_counts_by_class = {c:{con:0 for con in filtered_concepts} for c in configs.classes}
    
    num_act_images_saved = 0

    for i,con_row in concepts_df.iterrows():
        ch_row = channels_df.iloc[i]
        pred = con_row['pred']
        id = con_row['id']
        path = con_row['path']
        fname = con_row['file']
        act = acts_list[i]
        upact = upfn(torch.unsqueeze(act, dim=0))[0]
        image_channels_counts = image_channels_counts_list[i]
        filtered_image_channels_counts = {ch:image_channels_counts[ch] for ch in image_channels_counts if ch in filtered_channels}
        image_threshs = image_threshs_list[i]

        num_act_images = save_activation_images_of_image(iv, id, path, fname, upact, con_row, ch_row, filtered_image_channels_counts, 
                                                        channels_map, filtered_concepts, activation_images_path, image_threshs)
        num_act_images_saved += num_act_images

        for con in filtered_concepts:
            con_val = con_row[con]
            val = 1 if con_val == configs.high_value else 0
            filtered_concepts_counts[con] += val
            filtered_concepts_counts_by_class[pred][con] += val

        for ch in filtered_channels:
            val = ch_row[ch]
            filtered_channels_counts[ch] += 1 if val == configs.high_value else 0

    print('Saved {} activation images for {} images.'.format(num_act_images_saved, num_images))
    print('\nFiltered concept counts:', filtered_concepts_counts)
    for c,counts in filtered_concepts_counts_by_class.items():
        print('\nFiltered concept counts of class {}: {}'.format(c, counts))
    print('\nFiltered channel counts:', filtered_channels_counts)

    concepts_df.to_csv(concepts_output_path, index=False)
    channels_df.to_csv(channels_output_path, index=False)



def concept_attribution (dataset_path, model_file_path, result_path, concepts_file_path, channels_file_path, activation_images_path):
    model = load_model(model_file_path)
    model.retain_layer(configs.target_layer)

    dataset, data_loader = load_data(dataset_path)

    tally_path = os.path.join(result_path, 'report.json')
    thresholds_path = os.path.join(result_path, 'channel_quantiles.npy')
    channels_map, channels, concepts = load_channels_data(tally_path)

    args = EasyDict(model=configs.model_name, dataset=configs.dataset_name, seg=configs.seg_model_name, 
                    layer=configs.target_layer, quantile=configs.gradient_high_thresh)
    upfn = experiment.make_upfn(args, dataset, model, configs.target_layer)
    renorm = renormalize.renormalizer(dataset, target='zc')

    segmodel = None
    seg_concept_index_map = {}
    if configs.check_seg_overlap:
        segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(configs.seg_model_name)
        for i,lbl in enumerate(seglabels):
            seg_concept_index_map[lbl] = i

    model.stop_retaining_layers([configs.target_layer])

    concepts_df, channels_df, acts_list, image_channels_counts_list, image_threshs_list = \
        extract_concepts(model, segmodel, upfn, renorm, data_loader, channels_map, seg_concept_index_map, channels, concepts)

    if configs.filter_concepts:
        concepts_df, channels_df, concepts, channels = filter_extracted_concepts(concepts_df, channels_df, channels_map)

    save_image_concepts_dataset(concepts_df, channels_df, image_channels_counts_list, image_threshs_list, acts_list, 
        upfn, dataset, channels_map, concepts, channels, concepts_file_path, channels_file_path, activation_images_path)

