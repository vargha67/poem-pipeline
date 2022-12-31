import configs
from model_utils import vgg16_model
import torch, os, json, numpy
import IPython
from IPython.display import display
from torchvision import transforms, models
from new_dissection.netdissect import pbar, nethook, renormalize, parallelfolder, tally, imgviz, imgsave, show
from new_dissection.netdissect.easydict import EasyDict
from new_dissection.experiment import dissect_experiment as experiment



def load_model (model_file=None):
    if configs.model_name == 'vgg16':
        model = vgg16_model(num_classes=configs.num_classes)
    else:
        model = models.__dict__[configs.model_name](num_classes=configs.num_classes)
    checkpoint = torch.load(model_file)
    statedict = checkpoint
    if 'state_dict' in checkpoint:
        statedict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(statedict)
    model = nethook.InstrumentedModel(model).cuda().eval()

    return model



def load_dataset (dataset_dir=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(configs.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.norm_mean, std=configs.norm_std)
    ])

    dataset = parallelfolder.ParallelImageFolders([dataset_dir], classification=True, shuffle=True, transform=transform)
    print('Processing {} data examples from these classes: {}'.format(len(dataset), dataset.classes))

    return dataset



def show_sample_images (model, dataset, sample_batch, batch_indices, classlabels):
    truth = [classlabels[dataset[i][1]] for i in batch_indices]
    preds = model(sample_batch.cuda()).max(1)[1]
    imgs = [renormalize.as_image(t, source=dataset) for t in sample_batch]
    prednames = [classlabels[p.item()] for p in preds]
    show([[img, 'pred: ' + pred, 'true: ' + gt] for img, pred, gt in zip(imgs, prednames, truth)])



def show_sample_segmentations (segmodel, dataset, sample_batch, renorm):
    iv = imgviz.ImageVisualizer(120, source=dataset)
    seg = segmodel.segment_batch(renorm(sample_batch).cuda(), downsample=4)

    torch.set_printoptions(profile="full")
    print('seg.shape:', seg.shape)

    show([(iv.image(sample_batch[i]), iv.segmentation(seg[i,0]),
        iv.segment_key(seg[i,0], segmodel))
        for i in range(len(seg))])



def show_sample_heatmaps (model, dataset, sample_batch):
    acts = model.retained_layer(configs.target_layer).cpu()
    print('acts.shape:', acts.shape)
    print('acts_reshaped.shape:', acts.view(acts.shape[0], acts.shape[1], -1).shape)

    ivsmall = imgviz.ImageVisualizer((100, 100), source=dataset)
    display(show.blocks(
        [[[ivsmall.masked_image(sample_batch[0], acts, (0, u), percent_level=configs.activation_high_thresh)],
        [ivsmall.heatmap(acts, (0, u), mode='nearest')]] for u in range(min(acts.shape[1], 12))]
    ))



def show_sample_image_activation (model, dataset, rq, topk, classlabels, sample_unit_number, sample_image_index):
    print(topk.result()[1][sample_unit_number][sample_image_index], dataset.images[topk.result()[1][sample_unit_number][sample_image_index]])
    image_number = topk.result()[1][sample_unit_number][sample_image_index].item()

    iv = imgviz.ImageVisualizer((224, 224), source=dataset, quantiles=rq,
        level=rq.quantiles(configs.activation_high_thresh))
    batch = torch.cat([dataset[i][0][None,...] for i in [image_number]])
    truth = [classlabels[dataset[i][1]] for i in [image_number]]
    preds = model(batch.cuda()).max(1)[1]
    imgs = [renormalize.as_image(t, source=dataset) for t in batch]
    prednames = [classlabels[p.item()] for p in preds]
    acts = model.retained_layer(configs.target_layer)
    print('acts.shape:', acts.shape)
    print('acts_reshaped.shape:', acts.view(acts.shape[0], acts.shape[1], -1).shape)
    image_acts = acts[0,sample_unit_number].cpu().numpy()
    unit_quant = rq.quantiles(configs.activation_high_thresh)[sample_unit_number].item()
    print('number of activations higher than quantile {}: {}'.format(unit_quant, numpy.sum(image_acts > unit_quant)))

    show([[img, 'pred: ' + pred, 'true: ' + gt] for img, pred, gt in zip(imgs, prednames, truth)])
    show([[iv.masked_image(batch[0], acts, (0, sample_unit_number))]])
    show([[iv.heatmap(acts, (0, sample_unit_number), mode='nearest')]])



def save_top_channel_images (model, dataset, rq, topk, result_dir):
    pbar.descnext('unit_images')
    iv = imgviz.ImageVisualizer((100, 100), source=dataset, quantiles=rq,
            level=rq.quantiles(configs.activation_high_thresh))
    
    def compute_acts(image_batch, label_batch):
        image_batch = image_batch.cuda()
        _ = model(image_batch)
        acts_batch = model.retained_layer(configs.target_layer)
        return acts_batch

    unit_images = iv.masked_images_for_topk(
        compute_acts, dataset, topk, k=5, num_workers=2, pin_memory=True, 
        cachefile=os.path.join(result_dir, 'top5images.npz'))
    
    image_row_width = 5
    pbar.descnext('saving images')
    imgsave.save_image_set(unit_images, resfile('image/unit%d.jpg'),
        sourcefile=os.path.join(result_dir, 'top%dimages.npz' % image_row_width))
    
    return unit_images



def show_sample_channel_images (unit_images, sample_unit_numbers, unit_label_high=None):
    for u in sample_unit_numbers:
        if unit_label_high is None:
            print('unit %d' % u)
        else:
            print('unit %d, label %s, iou %.3f' % (u, unit_label_high[u][1], unit_label_high[u][3]))
        display(unit_images[u])



# Computes and keeps channel activations for all images in a way that any activation quantile for each channel can be computed easily
def compute_tally_quantile (model, dataset, upfn, sample_size, result_dir):
    pbar.descnext('rq')
    def compute_samples(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(configs.target_layer)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    rq = tally.tally_quantile(compute_samples, dataset,
                            sample_size=sample_size,
                            r=8192,
                            num_workers=2,
                            pin_memory=True,
                            cachefile=os.path.join(result_dir, 'rq.npz'))
    return rq



# Computes and keeps maximum of channel activations for all images, 
# so that the top k images with the highest maximum activation value can be identified for each channel
def compute_tally_topk (model, dataset, sample_size, result_dir):
    pbar.descnext('topk')
    def compute_image_max(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(configs.target_layer)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]
        return acts

    topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
            batch_size=50, num_workers=2, pin_memory=True,
            cachefile=os.path.join(result_dir, 'topk.npz'))
    return topk



# Computes the best concepts matching each channel based on IoUs between concept segmentations and channel activations
def compute_top_channel_concepts (model, segmodel, upfn, dataset, rq, seglabels, segcatlabels, sample_size, renorm, result_dir):
    # "level_high" was formerly named "level_at_99"
    # "condi_high" was formerly named "condi99"
    # "iou_high" was formerly named "iou_99"
    # "unit_label_high" was formerly named "unit_label_99"

    # Getting the target quantile values of channels: 
    level_high = rq.quantiles(configs.activation_high_thresh).cuda()[None,:,None,None]
    level_low = rq.quantiles(configs.activation_low_thresh).cuda()[None,:,None,None]
    
    # Computing the overlap between all the channel activations and all the image segmentations: 
    def compute_conditional_indicator(batch, *args):
        image_batch = batch.cuda()
        seg = segmodel.segment_batch(renorm(image_batch), downsample=4)
        _ = model(image_batch)
        acts = model.retained_layer(configs.target_layer)
        hacts = upfn(acts)
        iacts = (hacts > level_high).float() # indicator
        return tally.conditional_samples(iacts, seg)

    pbar.descnext('condi_high')
    condi_high = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,
            num_workers=10, pin_memory=True,
            cachefile=os.path.join(result_dir, 'condi_high.npz'))
    
    # Computing the IoU between each channel and all the concepts: 
    iou_high = tally.iou_from_conditional_indicator_mean(condi_high)

    # Identifying the concept with max IoU for each channel: 
    # unit_label_high = [
    #         (concept.item(), seglabels[concept], segcatlabels[concept], bestiou.item())
    #         for (bestiou, concept) in zip(*iou_high.max(0))]

    unit_label_high = []
    for i,row in enumerate(iou_high.t()):
        top_ious, top_concepts = row.topk(k=3)
        top_list = [(con.item(), seglabels[con], segcatlabels[con], iou.item()) for con,iou in zip(top_concepts, top_ious)]
        top_item = top_list[0]
        top_label = top_item[1]

        # Though not ideal, this is the best we can do to exclude concepts which are very similar to the dataset classes: 
        if (len(configs.excluded_concepts) > 0) and (top_label in configs.excluded_concepts):
            print('Channel {} top concepts: {}'.format(i, top_list))
            top_item = (0, '-', ('-','-'), 0.0)
            for j in range(1,len(top_list)):
                item = top_list[j]
                label = item[1]
                iou = item[3]
                if label not in configs.excluded_concepts:
                    top_item = item
                    break
            print('Because top concept {} is among the excluded concepts, concept {} with iou {} is selected for channel {}'
                .format(top_label, top_item[1], top_item[3], i))
        unit_label_high.append(top_item)

    label_list = [labelcat for concept, label, labelcat, iou in unit_label_high if iou > configs.min_iou]

    print(len(unit_label_high))
    print(unit_label_high)
    
    return unit_label_high, label_list, level_high, level_low



def save_final_data (unit_label_high, label_list, level_high, level_low, result_dir):
    display(IPython.display.SVG(experiment.graph_conceptcatlist(label_list)))
    experiment.save_conceptcat_graph(os.path.join(result_dir, 'concepts_high.svg'), label_list)

    print('level_high.shape:', level_high.shape)
    print('level_low.shape:', level_low.shape)

    high_quantiles = level_high.view(-1).cpu().numpy()
    low_quantiles = level_low.view(-1).cpu().numpy()

    print('high_quantiles.shape:', high_quantiles.shape)
    print('low_quantiles.shape:', low_quantiles.shape)

    experiment.dump_json_file(os.path.join(result_dir, 'report.json'), dict(
            header=dict(
                name='%s %s %s' % (configs.model_name, configs.dataset_name, configs.seg_model_name),
                image='concepts_high.svg'),
            units=[
                dict(image='image/unit%d.jpg' % u,
                    unit=u, iou=iou, label=label, cat=labelcat[1], high_thresh=float(high_quantiles[u]), low_thresh=float(low_quantiles[u]))
                for u, (concept, label, labelcat, iou)
                in enumerate(unit_label_high)])
            )
    
    experiment.copy_static_file('report.html', os.path.join(result_dir, 'report.html'))

    # print('level_high.shape:', level_high.shape)
    # quantiles = level_high.view(-1).cpu().numpy()
    # print('quantiles.shape:', quantiles.shape)
    numpy.save(os.path.join(result_dir, 'channel_quantiles.npy'), high_quantiles)

    print('Channel high quantiles:')
    for i,q in enumerate(list(high_quantiles)):
        print('{}: {}'.format(i,q))

    print('Channel low quantiles:')
    for i,q in enumerate(list(low_quantiles)):
        print('{}: {}'.format(i,q))
    


def concept_identification (dataset_path, model_file_path, result_path):
    model = load_model(model_file_path)
    model.retain_layer(configs.target_layer)

    dataset = load_dataset(dataset_path)
    classlabels = dataset.classes
    sample_size = len(dataset)

    print('Inspecting layer %s of model %s on dataset %s' % (configs.target_layer, configs.model_name, configs.dataset_name))
    print(model)

    args = EasyDict(model=configs.model_name, dataset=configs.dataset_name, seg=configs.seg_model_name, 
                    layer=configs.target_layer, quantile=configs.activation_high_thresh)
    upfn = experiment.make_upfn(args, dataset, model, configs.target_layer)
    renorm = renormalize.renormalizer(dataset, target='zc')
    segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(configs.seg_model_name)

    print('Segmentation labels:')
    for i,lbl in enumerate(seglabels):
        print('{}: {} from category {}'.format(i, lbl, segcatlabels[i]))

    batch_indices = [10, 20, 30, 40, 50, 60, 70, 80]
    batch = torch.cat([dataset[i][0][None,...] for i in batch_indices])
    show_sample_images(model, dataset, batch, batch_indices, classlabels)

    show_sample_segmentations(segmodel, dataset, batch, renorm)

    show_sample_heatmaps(model, dataset, batch)

    rq = compute_tally_quantile(model, dataset, upfn, sample_size, result_path)

    topk = compute_tally_topk(model, dataset, sample_size, result_path)

    show_sample_image_activation(model, dataset, rq, topk, classlabels, sample_unit_number=2, sample_image_index=0)

    unit_images = save_top_channel_images(model, dataset, rq, topk, result_path)
    sample_unit_numbers = [10, 20, 30, 40]
    show_sample_channel_images(unit_images, sample_unit_numbers)

    unit_label_high, label_list, level_high, level_low = compute_top_channel_concepts(model, segmodel, upfn, 
        dataset, rq, seglabels, segcatlabels, sample_size, renorm, result_path)

    show_sample_channel_images(unit_images, sample_unit_numbers, unit_label_high)

    save_final_data(unit_label_high, label_list, level_high, level_low, result_path)
