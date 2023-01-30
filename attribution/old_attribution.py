import configs
from model_utils import vgg16_model
from data_utils import CustomImageFolder
import torch, os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import pil_loader
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from new_dissection.netdissect import nethook, renormalize, imgviz
from new_dissection.netdissect.easydict import EasyDict
from new_dissection.experiment import dissect_experiment as experiment



def load_model (model_file):
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
    data_loader = DataLoader(dataset, batch_size=configs.batch_size)
    print('Processing {} data examples in {} batches with class index {}'.format(len(dataset), len(data_loader), dataset.class_to_idx))
    
    return dataset, data_loader



def load_channels_data (tally_path, thresholds_path):
    tally_data = pd.read_csv(tally_path)
    tally_data = tally_data[tally_data['score'] > configs.min_iou]

    channels_list = tally_data['unit'].tolist()
    concepts_list = tally_data['label'].tolist()
    categories_list = tally_data['category'].tolist()

    channels_map = {}
    for ch, con, cat in zip(channels_list, concepts_list, categories_list):
        channels_map[ch] = {
            'concept': con,
            'category': cat
        }

    channels = list(set(channels_list))
    concepts = list(set(concepts_list))

    channels.sort()
    concepts.sort()

    thresholds = np.load(thresholds_path)
    for i,t in enumerate(thresholds):
        if i+1 in channels_map:
            channels_map[i+1]['thresh'] = t

    print('Processing {} concepts and {} channels'.format(len(concepts), len(channels)))
    print('channels_map:', channels_map)

    return channels_map, channels, concepts



def extract_concepts_from_image (act, upact, seg, channels_map, seg_concept_index_map, channels, concepts):
    num_channels = act.shape[0]
    image_concepts = {con:0 for con in concepts}  # used to hold the value of each concept for this image (used for image concepts file and saving activation images process)
    image_channels = {ch:0 for ch in channels}    # used to hold the value of each channel for this image (used for image channels file and saving activation images process)
    image_concepts_counts = {con:0 for con in concepts}   # used to count how many channels related to each concept were high for this image (just used for stat keeping purposes)
    image_channels_counts = {ch:0 for ch in channels}     # used to keep the number of high thresh pixels for each channel for this image (used in saving activation images process)
    image_overlap_ratio = 0
    n_channels_attributed = 0

    for ch in channels:
        ch_index = ch - 1
        ch_upact = upact[ch_index]
        ch_info = channels_map[ch]
        channel_concept = ch_info['concept']
        channel_category = ch_info['category']
        channel_thresh = ch_info['thresh']

        if (ch_upact is None) or (channel_concept is None) or (channel_category is None) or (channel_thresh is None):
            continue

        ch_upact_high_mask = (ch_upact > channel_thresh)
        num_high_thresh = np.sum(ch_upact_high_mask.numpy())

        if num_high_thresh >= configs.min_thresh_pixels:
            image_channels[ch] = 1
            image_concepts[channel_concept] = 1
            image_concepts_counts[channel_concept] += 1
            image_channels_counts[ch] = num_high_thresh

            if configs.check_seg_overlap:
                seg_concept_index = seg_concept_index_map[channel_concept] if channel_concept in seg_concept_index_map else None
                if seg_concept_index is None:
                    # Handling common cases where the channel concept is not found in the segmentation concepts index, 
                    # but a similar concept may exist:
                    alt_concept = None
                    if channel_concept.endswith('-c'):
                        alt_concept = channel_concept[:-2]
                    else:
                        alt_concepts_map = {
                            'windowpane': 'window'
                        }
                        alt_concept = alt_concepts_map[channel_concept] if channel_concept in alt_concepts_map else None

                    if alt_concept != None: 
                        seg_concept_index = seg_concept_index_map[alt_concept] if alt_concept in seg_concept_index_map else None

                cat_index = configs.category_index_map[channel_category] if channel_category in configs.category_index_map else None
                if (seg_concept_index is None) or (cat_index is None):
                    print('Error: Missing segmentation concept index ({}) or category index ({}) for channel {} mapped to concept {} from category {}'
                        .format(seg_concept_index, cat_index, ch, channel_concept, channel_category))
                else:
                    n_channels_attributed += 1
                    target_seg = seg[cat_index]
                    target_seg_concept_mask = (target_seg == seg_concept_index)
                    num_concept_seg = np.sum(target_seg_concept_mask.numpy())
                    overlap_mask = ch_upact_high_mask & target_seg_concept_mask
                    num_overlap = np.sum(overlap_mask.numpy())

                    if configs.overlap_mode == 'overlap_to_union_ratio':
                        union_mask = ch_upact_high_mask | target_seg_concept_mask
                        num_union = np.sum(union_mask.numpy())
                        overlap_ratio = num_overlap / num_union
                        image_overlap_ratio += overlap_ratio

                    elif configs.overlap_mode == 'overlap_to_activation_ratio':
                        overlap_ratio = num_overlap / num_high_thresh
                        image_overlap_ratio += overlap_ratio

                    elif configs.overlap_mode == 'overlap_to_segmentation_ratio':
                        overlap_ratio = num_overlap / num_concept_seg
                        image_overlap_ratio += overlap_ratio

    if n_channels_attributed > 0:
        image_overlap_ratio = image_overlap_ratio / n_channels_attributed
        #print('image_overlap_ratio:', image_overlap_ratio)
    return image_concepts, image_channels, image_concepts_counts, image_channels_counts, image_overlap_ratio, n_channels_attributed



def extract_concepts (model, segmodel, upfn, renorm, data_loader, channels_map, seg_concept_index_map, channels, concepts):
    activations = []

    def activations_hook (module, input, output):
        activations.append(output.detach().cpu())

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
    acts_list = []
    num_images = 0
    num_images_attributed = 0
    total_acc = 0
    total_overlap_ratio = 0
    
    with torch.no_grad():
        for i, (images, labels, paths) in tqdm(enumerate(data_loader)):
            del activations[:]
            images_gpu = images.cuda()
            labels_gpu = labels.cuda()
        
            output = model(images_gpu)

            _, preds = torch.max(output, 1)
            acts = activations[0]

            total_acc += (preds == labels_gpu).float().sum()
            preds = preds.cpu().numpy()
            labels = labels.numpy()

            upacts = upfn(acts)
            segs = None
            if configs.check_seg_overlap:
                segs = segmodel.segment_batch(renorm(images_gpu), downsample=4).cpu()

            for j in range(images.shape[0]):
                num_images += 1
                pred = preds[j]
                label = labels[j]
                path = paths[j]
                image = images[j]
                act = acts[j]
                upact = upacts[j]
                seg = segs[j] if segs != None else None
                _, fname = os.path.split(path)

                image_concepts, image_channels, image_concepts_counts, image_channels_counts, image_overlap_ratio, n_channels_attributed = \
                    extract_concepts_from_image(act, upact, seg, channels_map, seg_concept_index_map, channels, concepts)

                acts_list.append(act)
                image_channels_counts_list.append(image_channels_counts)

                if n_channels_attributed > 0:
                    num_images_attributed += 1
                    total_overlap_ratio += image_overlap_ratio

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
    total_overlap_ratio = total_overlap_ratio / num_images_attributed
    print('\nExtracted concepts from {} total and {} attributed images with accuracy {:.3f} and overlap ratio {:.2f}.' \
        .format(num_images, num_images_attributed, total_acc, total_overlap_ratio))
    print('\nConcept counts:', concepts_counts)
    for c,counts in concepts_counts_by_class.items():
        print('\nConcept counts of class {}: {}'.format(c, counts))
    print('\nChannel counts:', channels_counts)

    concepts_df = pd.DataFrame(concepts_rows_list)
    channels_df = pd.DataFrame(channels_rows_list)

    return concepts_df, channels_df, acts_list, image_channels_counts_list, total_overlap_ratio



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



def save_activation_images_of_image (iv, image_index, image_path, image_fname, acts, image_channels_counts, 
                                     channels_map, concepts, output_dir):
    image_activated_channels = [k for k,v in image_channels_counts.items() if v > 0]   # Only keep those channels which have been high for the image
    if len(image_activated_channels) == 0:
        # print('Image {} with path {} has no activated channels!'.format(image_index, image_path))
        return 0

    acts = acts[None, :, :, :]   # as required by iv.masked_image

    image_concept_channels = {con:[] for con in concepts}
    for ch in image_activated_channels:
        ch_info = channels_map[ch]
        channel_concept = ch_info['concept']
        num_high_thresh = image_channels_counts[ch]   # In case of binning features, it can be the count of either mid or high pixels, depending on whether the channel has been mid or high for the image

        if (channel_concept is None) or (num_high_thresh is None):
            #print('Error: Missing concept ({}) or number of high-thresh pixels ({}) for channel {}!'.format(channel_concept, num_high_thresh, ch))
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

    num_act_images = 0
    for con,lst in image_concept_channels.items():
        if len(lst) == 0:
            continue

        top_channel_nums = sorted(lst, key=lambda x: x[1], reverse=True)[:configs.n_top_channels_per_concept]
        top_channels = [k for k,v in top_channel_nums]

        for i,ch in enumerate(top_channels):
            ch_index = ch - 1
            ch_info = channels_map[ch]
            channel_thresh = ch_info['thresh']

            if (channel_thresh is None):
                print('Error: Missing activation or threshold ({}) for channel {}!'.format(channel_thresh, ch))
                continue

            new_image = iv.masked_image(image, acts, (0, ch_index), level=channel_thresh)

            ind = image_fname.rfind('.')
            image_fname_raw = image_fname[:ind]
            new_fname = image_fname_raw + '_' + str(ch) + '_' + con + '.jpg'
            new_path = os.path.join(output_dir, new_fname)

            new_image.save(new_path, optimize=True, quality=99)
            num_act_images += 1

    # print('Saved {} activation images for image {} with path {}'.format(num_act_images, image_index, image_path))
    return num_act_images



def save_image_concepts_dataset (concepts_df, channels_df, image_channels_counts_list, total_overlap_ratio, 
                                 acts_list, upfn, dataset, channels_map, filtered_concepts, filtered_channels, 
                                 concepts_output_path, channels_output_path, activation_images_path, concepts_evaluation_file_path):
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

        num_act_images = save_activation_images_of_image(iv, id, path, fname, upact, filtered_image_channels_counts, 
                                                         channels_map, filtered_concepts, activation_images_path)
        num_act_images_saved += num_act_images

        for con in filtered_concepts:
            val = con_row[con]
            filtered_concepts_counts[con] += val
            filtered_concepts_counts_by_class[pred][con] += val

        for ch in filtered_channels:
            val = ch_row[ch]
            filtered_channels_counts[ch] += val

    print('Saved {} activation images for {} images.'.format(num_act_images_saved, num_images))
    print('\nFiltered concept counts:', filtered_concepts_counts)
    for c,counts in filtered_concepts_counts_by_class.items():
        print('\nFiltered concept counts of class {}: {}'.format(c, counts))
    print('\nFiltered channel counts:', filtered_channels_counts)

    concepts_df.to_csv(concepts_output_path, index=False)
    channels_df.to_csv(channels_output_path, index=False)

    original_concepts = list(set([v['concept'] for k,v in channels_map.items()]))
    original_concepts.sort()

    eval_results = {
        'concepts': original_concepts,
        'num_concepts': len(original_concepts),
        'filtered_concepts': filtered_concepts,
        'num_filtered_concepts': len(filtered_concepts),
        'avg_overlap_ratio': total_overlap_ratio
    }

    with open(concepts_evaluation_file_path, 'w') as f:
        json.dump(eval_results, f, indent=4)



def concept_attribution (dataset_path, model_file_path, result_path, concepts_file_path, 
                         channels_file_path, activation_images_path, concepts_evaluation_file_path):
    model = load_model(model_file_path)
    model.retain_layer(configs.target_layer)

    dataset, data_loader = load_data(dataset_path)

    tally_path = os.path.join(result_path, 'tally.csv')
    thresholds_path = os.path.join(result_path, 'quantile.npy')
    channels_map, channels, concepts = load_channels_data(tally_path, thresholds_path)

    args = EasyDict(model=configs.model_name, dataset=configs.dataset_name, seg=configs.seg_model_name, 
                    layer=configs.target_layer, quantile=configs.activation_high_thresh)
    upfn = experiment.make_upfn(args, dataset, model, configs.target_layer)
    renorm = renormalize.renormalizer(dataset, target='zc')

    segmodel = None
    seg_concept_index_map = {}
    if configs.check_seg_overlap:
        segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(configs.seg_model_name)
        for i,lbl in enumerate(seglabels):
            seg_concept_index_map[lbl] = i

    model.stop_retaining_layers([configs.target_layer])

    concepts_df, channels_df, acts_list, image_channels_counts_list, total_overlap_ratio = \
        extract_concepts(model, segmodel, upfn, renorm, data_loader, channels_map, seg_concept_index_map, channels, concepts)

    if configs.filter_concepts_old:
        concepts_df, channels_df, concepts, channels = filter_extracted_concepts(concepts_df, channels_df, channels_map)

    save_image_concepts_dataset(concepts_df, channels_df, image_channels_counts_list, total_overlap_ratio, 
        acts_list, upfn, dataset, channels_map, concepts, channels, concepts_file_path, 
        channels_file_path, activation_images_path, concepts_evaluation_file_path)
