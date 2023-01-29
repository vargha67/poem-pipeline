import configs
from model_utils import vgg16_model
from data_utils import CustomImageFolder
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from PIL import Image
from imageio import imwrite
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models



def load_model (model_path):
    if configs.model_name == 'vgg16':
        model = vgg16_model(num_classes=configs.num_classes)
    else:
        model = models.__dict__[configs.model_name](num_classes=configs.num_classes)
    model.load_state_dict(torch.load(model_path))

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
    
    return data_loader



def load_channels_data (tally_path, thresholds_path):
    tally_data = pd.read_csv(tally_path)
    tally_data = tally_data[tally_data['score'] > configs.min_iou]

    channels_list = tally_data['unit'].tolist()
    concepts_list = tally_data['label'].tolist()

    channel_concept_map = {k:v for k,v in zip(channels_list, concepts_list)}
    channels = list(set(channels_list))
    concepts = list(set(concepts_list))

    channels.sort()
    concepts.sort()

    thresholds = np.load(thresholds_path)
    channel_thresh_map = {i+1:t for i,t in enumerate(thresholds) if i+1 in channels}

    print('Processing {} concepts and {} channels'.format(len(concepts), len(channels)))
    print('channel_concept_map:', channel_concept_map)
    print('channel_thresh_map:', channel_thresh_map)

    return channel_concept_map, channel_thresh_map, channels, concepts



def extract_concepts_from_batch (acts, channel_concept_map, channel_thresh_map, channels, concepts):
    batch_concepts_counts = []
    batch_channels_counts = []
    num_images = acts.shape[0]
    
    for i in range(num_images):
        act = acts[i]
        num_channels = act.shape[0]
        image_concepts_counts = {con:0 for con in concepts}
        image_channels_counts = {ch:0 for ch in channels}

        for ch in channels:
            ch_index = ch - 1
            channel_activation = act[ch_index]
            channel_thresh = channel_thresh_map[ch] if ch in channel_thresh_map else None
            channel_concept = channel_concept_map[ch] if ch in channel_concept_map else None

            if (channel_activation is None) or (channel_thresh is None) or (channel_concept is None):
                print('Error: Missing activation, concept ({}), or threshold ({}) for channel {}!'.format(channel_concept, channel_thresh, ch))
                continue

            num_high_thresh = np.sum(channel_activation > channel_thresh)
            if num_high_thresh >= configs.min_thresh_pixels_old:
                image_concepts_counts[channel_concept] += 1
                image_channels_counts[ch] = num_high_thresh
                if ch not in channels:
                    print('Error: channel {} not among channels list!'.format(ch))

        batch_concepts_counts.append(image_concepts_counts)
        batch_channels_counts.append(image_channels_counts)

    return batch_concepts_counts, batch_channels_counts



def extract_concepts (model, data_loader, channel_concept_map, channel_thresh_map, channels, concepts):
    batch_activations = []

    def get_activations(module, input, output):
        batch_activations.append(output.data.cpu().numpy())

    layer_names = [configs.target_layer]
    for name in layer_names:
        #model._modules.get(name).register_forward_hook(get_activations)
        layer = model._modules.get(name)
        if layer is None:
            for n,l in model.named_modules():
                if n == name:
                    layer = l
                    print('Target layer found:', n)
                    break
        layer.register_forward_hook(get_activations)

    model.eval()

    concepts_counts = {con:0 for con in concepts}
    channels_counts = {ch:0 for ch in channels}
    concepts_counts_by_class = {c:{con:0 for con in concepts} for c in configs.classes}

    concepts_rows_list = []
    channels_rows_list = []
    image_channels_counts_list = []
    acts_list = []
    num_images = 0
    total_acc = 0
    
    with torch.no_grad():
        for i, (images, labels, paths) in tqdm(enumerate(data_loader)):
            del batch_activations[:]
            images_gpu = images.cuda()
            labels_gpu = labels.cuda()
        
            output = model(images_gpu)

            _, preds = torch.max(output, 1)
            total_acc += (preds == labels_gpu).float().sum()
            preds = preds.cpu().numpy()
            labels = labels.numpy()
            images = images.numpy()
            acts = batch_activations[0]   # currently assume there is only one target layer

            # if i == 0:
            #     print('images: {}, labels: {}, preds: {}'.format(images.shape, labels.shape, preds.shape))
            #     print('batch_activations.shape: {} * {}'.format(len(batch_activations), batch_activations[0].shape))
            #     print('paths:', paths)

            batch_concepts_counts, batch_channels_counts = \
                extract_concepts_from_batch(acts, channel_concept_map, channel_thresh_map, channels, concepts)

            for j in range(len(batch_concepts_counts)):
                num_images += 1
                image_concepts_counts = batch_concepts_counts[j]
                image_channels_counts = batch_channels_counts[j]
                pred = preds[j]
                label = labels[j]
                path = paths[j]
                image = images[j]
                act = acts[j]
                _, fname = os.path.split(path)

                # if i == 0 and j == 0:
                #     print('Image {} with label {}, pred {}, act shape {}, and concepts counts {}'
                #         .format(path, label, pred, act.shape, image_concepts_counts))
                    
                acts_list.append(act)
                image_channels_counts_list.append(image_channels_counts)

                image_concepts_row = {k:1 if v > 0 else 0 for k,v in image_concepts_counts.items()}
                image_concepts_row['pred'] = pred
                image_concepts_row['label'] = label
                image_concepts_row['id'] = num_images
                image_concepts_row['file'] = fname
                image_concepts_row['path'] = path
                concepts_rows_list.append(image_concepts_row)

                image_channels_row = {k:1 if v > 0 else 0 for k,v in image_channels_counts.items()}
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

    return concepts_df, channels_df, acts_list, image_channels_counts_list



def filter_extracted_concepts (concepts_df, channels_df, channel_concept_map):
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
    filtered_channels = [ch for ch in channel_cols if (ch in channel_concept_map) and (channel_concept_map[ch] in filtered_concepts)]
    print('Channels reduced from {} to {} by concept filtering.'.format(len(channel_cols), len(filtered_channels)))

    cols_to_keep = filtered_channels + meta_cols
    filtered_channels_df = channels_df[cols_to_keep]
    # display(filtered_channels_df.head())

    return filtered_concepts_df, filtered_channels_df, filtered_concepts, filtered_channels



def save_activation_images (image_index, image_path, image_fname, acts, image_channels_counts, channel_concept_map, 
                            channel_thresh_map, concepts, output_dir):
    image_activated_channels = [k for k,v in image_channels_counts.items() if v > 0]
    if len(image_activated_channels) == 0:
        # print('Image {} with path {} has no activated channels!'.format(image_index, image_path))
        return 0

    # # Un-normalizing the image back to its original form: 
    # torch_image = torch.from_numpy(image)
    # torch_image.mul_(torch.as_tensor(norm_std).view(-1,1,1)).add_(torch.as_tensor(norm_mean).view(-1,1,1))   # normalization actually did the reverse: torch_image.sub_(norm_mean).div_(norm_std)
    # image = torch_image.numpy()

    # Preferred to reopen the image and apply the initial resize transform to it without the normalization step, instead of manual un-normalization:
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(configs.image_size),
        transforms.ToTensor()
    ])
    image = transform(image).numpy()

    # Changing the shape of image from CxHxW to HxWxC format: 
    image = np.transpose(image, (1, 2, 0))   # image.permute(1, 2, 0) in PyTorch

    # Normalizing the pixel values to (0, 255) range required later for saving the image: 
    image_min, image_max = np.min(image), np.max(image)
    image = (((image - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
    # if image_index in [1,2]:
    #     print('image_min: {}, image_max: {}, new_image.min: {}, new_image.max: {}'.format(image_min, image_max, np.min(image), np.max(image)))

    image_concept_channels = {con:[] for con in concepts}
    for ch in image_activated_channels:
        channel_concept = channel_concept_map[ch]
        num_high_thresh = image_channels_counts[ch]

        if (channel_concept is None) or (num_high_thresh is None):
            #print('Error: Missing concept ({}) or number of high-thresh pixels ({}) for channel {}!'.format(channel_concept, num_high_thresh, ch))
            continue
            
        image_concept_channels[channel_concept].append((ch, num_high_thresh))

    cnt = 0
    for con,lst in image_concept_channels.items():
        if len(lst) == 0:
            continue

        top_channel_nums = sorted(lst, key=lambda x: x[1], reverse=True)[:configs.n_top_channels_per_concept]
        top_channels = [k for k,v in top_channel_nums]
        # if image_index in [1,2]:
        #     print('Top channels for concept {}: {}'.format(con, top_channel_nums))

        for i,ch in enumerate(top_channels):
            ch_index = ch - 1
            channel_activation = acts[ch_index]
            channel_thresh = channel_thresh_map[ch]
            channel_concept = con

            if (channel_activation is None) or (channel_thresh is None):
                print('Error: Missing activation or threshold ({}) for channel {}!'.format(channel_thresh, ch))
                continue

            should_print = (image_index in [1,2]) and (i == 0)

            mask = np.array(Image.fromarray(channel_activation).resize(size=(image.shape[1], image.shape[0]), resample=Image.BILINEAR))   # size=image.shape[:2]
            mask = mask > channel_thresh
            new_image = (mask[:, :, np.newaxis] * configs.overlay_opacity + (1 - configs.overlay_opacity)) * image

            ind = image_fname.rfind('.')
            image_fname_raw = image_fname[:ind]
            new_fname = image_fname_raw + '_' + str(ch) + '_' + channel_concept + '.jpg'
            new_path = os.path.join(output_dir, new_fname)

            final_image = new_image.astype(np.uint8)
            # if should_print:
            #     print('new_image.min: {}, new_image.max: {}, final_image.min: {}, final_image.max: {}'
            #         .format(np.min(new_image), np.max(new_image), np.min(final_image), np.max(final_image)))

            imwrite(new_path, final_image)
            cnt += 1
    
    # print('Saved {} activation images for image {} with path {}'.format(cnt, image_index, image_path))
    return cnt



def save_image_concepts_dataset (concepts_df, channels_df, image_channels_counts_list, acts_list, channel_concept_map, channel_thresh_map, 
                                filtered_concepts, filtered_channels, concepts_output_path, channels_output_path, activation_images_path):
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
        image_channels_counts = image_channels_counts_list[i]
        filtered_image_channels_counts = {ch:image_channels_counts[ch] for ch in image_channels_counts if ch in filtered_channels}

        for con in filtered_concepts:
            val = con_row[con]
            filtered_concepts_counts[con] += val
            filtered_concepts_counts_by_class[pred][con] += val

        for ch in filtered_channels:
            val = ch_row[ch]
            filtered_channels_counts[ch] += val

        num_act_images = save_activation_images(id, path, fname, act, filtered_image_channels_counts, channel_concept_map, channel_thresh_map, 
                                                filtered_concepts, activation_images_path)
        num_act_images_saved += num_act_images

    print('Saved {} activation images for {} images.'.format(num_act_images_saved, num_images))
    print('\nFiltered concept counts:', filtered_concepts_counts)
    for c,counts in filtered_concepts_counts_by_class.items():
        print('\nFiltered concept counts of class {}: {}'.format(c, counts))
    print('\nFiltered channel counts:', filtered_channels_counts)

    concepts_df.to_csv(concepts_output_path, index=False)
    channels_df.to_csv(channels_output_path, index=False)



def concept_attribution (dataset_path, model_file_path, result_path, concepts_file_path, channels_file_path, activation_images_path):
    data_loader = load_data(dataset_path)

    tally_path = os.path.join(result_path, 'tally.csv')
    thresholds_path = os.path.join(result_path, 'quantile.npy')
    channel_concept_map, channel_thresh_map, channels, concepts = load_channels_data(tally_path, thresholds_path)

    model = load_model(model_file_path)
    model = model.cuda()

    concepts_df, channels_df, acts_list, image_channels_counts_list = \
        extract_concepts(model, data_loader, channel_concept_map, channel_thresh_map, channels, concepts)

    if configs.filter_concepts_old:
        concepts_df, channels_df, concepts, channels = filter_extracted_concepts(concepts_df, channels_df, channel_concept_map)

    save_image_concepts_dataset(concepts_df, channels_df, image_channels_counts_list, acts_list, channel_concept_map, 
        channel_thresh_map, concepts, channels, concepts_file_path, channels_file_path, activation_images_path)
