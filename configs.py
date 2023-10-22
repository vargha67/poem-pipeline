from utils import extract_class_titles

# CNN models available to be used: 
model_settings = {
    "resnet18": {
        'target_layer': 'layer4.1.conv2',   # title of the target convolutional layer of the model to be used for explanation
        'url': 'https://drive.google.com/file/d/1PFztt2odOnxsih7FSxrAEb-g7oUlPZcg/view?usp=share_link'
    }, 
    "resnet50": {
        'target_layer': 'layer4.2.conv3',
        'url': 'https://drive.google.com/file/d/1it6kuferC3VmR7YGdnUzTgXwjU-2tBco/view?usp=share_link'
    }, 
    "vgg16": {
        'target_layer': 'features.conv5_3',   # 'features.28'
        'url': 'https://drive.google.com/file/d/1r3wO_FWnSKIHI34B8HiZjOyxFW8dk1ZV/view?usp=share_link'
    },
    # "alexnet": {
    #     'target_layer': 'conv5'
    # }
}

# Target dataset-class configurations available to be used: 
dataset_settings = {
    "places_bedroom_kitchen_livingroom": {   # title includes the main name of the dataset, followed by the name of the target classes
        'load_model': True,   # whether the pretrained model needs to be loaded from url/file, or is available by default in Torchvision models
        'num_classes': 3,   # number of target classes to be used for explanation
        'base_num_classes': 365,   # original number of classes available in the dataset
        'url': 'https://drive.google.com/file/d/1mVHRa1tDtUyW3dSdOCmGPPAYMjFR-8l4/view?usp=share_link'
    },
    "places_coffeeshop_restaurant": {
        'load_model': True, 
        'num_classes': 2, 
        'base_num_classes': 365, 
        'url': 'https://drive.google.com/file/d/1E4f2zVkTwwQI4OBQeGeGedkWS4w0cCUq/view?usp=share_link'
    },
    "imagenet_minivan_pickup": {
        'load_model': False, 
        'num_classes': 2, 
        'base_num_classes': 1000, 
        'excluded_concepts': ['pickup', 'van'],   # ['car', 'bus', 'coach', 'truck', 'van']
        'url': 'https://drive.google.com/file/d/1mPJi4iLMsSQb8P145t_LtWyA2MVZtJ23/view?usp=share_link'
    },
    "imagenet_laptop_mobile": {
        'load_model': False, 
        'num_classes': 2, 
        'base_num_classes': 1000, 
        'excluded_concepts': ['laptop', 'mobile', 'computer'],
        'url': 'https://drive.google.com/file/d/1ACPq3T0vy_Dm9Twzu3cy824IcKDcQrd9/view?usp=share_link'
    }
}

# Run configuration: 
run_pretraining = True   # run the model finetuning step or not
run_identification = True   # run the concept identification step or not
run_attribution = True   # run the concept attribution step or not
run_pattern_mining = True   # run the pattern mining step or not
run_evaluation = True   # run the method performance and runtime evaluation or not
run_visualization = True   # run the sample visualizations or not

# Model and dataset settings: 
model_name = 'resnet18'   # the CNN model name to be used (should be available in model_settings)
dataset_name = 'places_bedroom_kitchen_livingroom'   # the dataset-class config to be used (should be available in dataset_settings)
dataset_pure_name = dataset_name.split('_')[0]   # the dataset main name extracted from the dataset-class config (e.g. Places)

current_model_setting = model_settings[model_name]
target_layer = current_model_setting['target_layer']
base_model_url = current_model_setting['url']

current_dataset_setting = dataset_settings[dataset_name] 
load_model_from_disk = current_dataset_setting['load_model']
num_classes = current_dataset_setting['num_classes']
base_num_classes = current_dataset_setting['base_num_classes']
full_dataset_url = current_dataset_setting['url']

broden_dataset_url = 'https://drive.google.com/file/d/1U96rDGgarZwL6bX7RaStTY_wT30MLusD/view?usp=share_link'   # url of the Broden dataset used in CNN2DT
segmenter_model_url = 'https://drive.google.com/file/d/1DTNvF80el488egl2s9uaz7uIbA5La8sa/view?usp=share_link'   # url of the segmentation model

# Pretraining/finetuning settings: 
pretrain_mode = 'feature_extraction'   # type of finetuning applied on the model: 'full_fine_tuning', 'partial_fine_tuning', 'feature_extraction'
target_classes = []   # in case the dataset url includes all the original classes, and we need to select a subset for explanation
train_ratio = 0.7   # ratio of data used for finetuning of the model
explanation_ratio = 0.3   # ratio of data used for explanation of the model (should be <= 1 - train_ratio)
train_batch_size = 32   # batch size used for finetuning
epochs = 50   # number of epochs used for finetuning
stop_patience = 5   # number of epochs without improvement to stop finetuning after
cnn_dropout = 0.0   # dropout ratio used after CNN layers of the model (recommended to be zero for explanation)
dense_dropout = 0.0   # dropout ratio used after dense layers of the model (recommended to be zero for explanation)
random_seed = 1   # random seed used to ensure reproducibility of model finetuning, data split, etc. 

# Preprocessing settings: 
image_size = 224   # target width/height of the images
norm_mean = (0.485, 0.456, 0.406)   # mean coefficients used for normalizing the images
norm_std = (0.229, 0.224, 0.225)   # std coefficients used for normalizing the images

# Concept identification settings:
old_process = False   # whether to use CNN2DT (true) or POEM (false) for the entire pipeline
seg_model_name = 'netpc'   # type of the segmentation model in terms of details: 'netpc' or 'netpqc'
min_iou = 0.04   # min intersection-over-union to map a filter to a concept it focuses on
activation_high_thresh = 0.99   # min threshold to decide the highly-activated areas of a filter activation map
activation_low_thresh = 0.7   # min threshold to decide the mid and low-activated areas of a filter activation map (only used in case of binning_features enabled)
exclude_similar_concepts = True   # it is better to exclude general concepts which are similar to the dataset classes; e.g. laptop or computer concepts in laptop vs mobile dataset
excluded_concepts = current_dataset_setting['excluded_concepts'] if exclude_similar_concepts and ('excluded_concepts' in current_dataset_setting) else []

# Concept attribution settings:
batch_size = 16   # number of images to process in one batch
gradient_high_thresh = 0.95   # min threshold to decide the high-value areas of a activation-gradient map
min_thresh_pixels = 10   # min number of highly-activated pixels to allow concept attribution for an image in POEM (higher than CNN2DT because of upsampling)
n_top_channels_per_concept = 3   # max number of filters related to each concept to keep their activation images
check_gradients = True   # whether to check the gradients in addition to activations for concept attribution
pool_gradients = False   # whether to pool/average the gradients before multiplying by the activations
included_categories = ['object', 'material', 'part', 'color']   # categories of concepts to include in concept attribution

# Segment-gradient overlap settings:
check_seg_overlap = True   # whether to check segmentation overlap with activation-gradient maps as a criteria for concept attribution
overlap_mode = 'overlap_to_activation_ratio'   # type of overlap to check between segmentation and activation-gradient maps: 'overlap_pixels_count', 'overlap_to_union_ratio', 'overlap_to_activation_ratio', 'overlap_to_segmentation_ratio'
min_overlap_ratio = 0.5   # min ratio of overlap between segmentation and activation-gradient maps based on the overlap mode (e.g. 50% ratio of overlap to high activation area)
min_overlap_pixels = 5   # min number of pixels of overlap (only used in case of 'overlap_pixels_count' type)
category_index_map = {   # indices for segmentation/concept categories
    'object': 0,
    'material': 1,
    'part': 2,
    'color': 3
}

# Concept filtering settings: 
filter_concepts_old = True   # whether to filter out weak concepts in CNN2DT (which is not part of the original CNN2DT)
filter_concepts = True   # whether to filter out weak concepts in POEM
low_variance_thresh = 0.99   # concepts attributed to less than (1 - low_variance_thresh) of images are filtered out
max_concepts = 10   # max number of concepts to keep even after low variance filtering

# Binning features settings:
binning_features = False   # whether to bin concept features to high, mid, and low values instead of the default binary 0/1 case
gradient_low_thresh = 0.7   # mid threshold to decide the mid and low areas of the activation-gradient map (only used if binning_features is enabled)
high_value = 2 if binning_features else 1   # value used to represent the high value of concept features
mid_value = 1   # value used to represent the mid value of concept features (only used if binning_features is enabled)
low_value = 0   # value used to represent the low value of concept features

# Binning classes settings:
binning_classes = False   # whether to bin classes into highly and lowly-certain based on level of uncertainty of the CNN model in its decisions
certainty_thresh = 0.6   # min threshold of the model logits to decide the highly-certain class

# Pattern mining settings: 
remove_inactivated_patterns = False   # whether to exclude patterns having low-value concept features (i.e. patterns with some concepts equal to 0) from the list of generated patterns
class_titles = extract_class_titles(dataset_name, binning_classes)
classes = list(class_titles.keys())
class_names = list(class_titles.values())

rule_methods = ['cart'] if old_process else ['cart', 'exp', 'ids']   # rule mining methods to be used for explanation
min_support_params = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]   # min support params of rule mining methods to be tested and used for evaluation
ids_smooth_search = False   # whether to use smooth search instead of deterministic search for IDS
ids_timeout = 500000   # Timeout to stop IDS to avoid very long and useless running time

# Pattern visualization settings:
meta_cols = ['index', 'pred', 'support', 'confidence', 'accuracy', 'method', 'score']   # list of metadata/stats columns of patterns displayed
ids_param = 0.01   # min support used for visualization of IDS patterns
cart_param = 0.03   # min support used for visualization of CART patterns
exp_param = 0.03   # min support used for visualization of Exp patterns
max_patterns = 10   # max number of patterns to display

# Evaluation settings: 
max_patterns_to_keep = [5 for sup in min_support_params]   # max number of patterns to keep for each rule mining method to use for evaluation



def update_configs(_run_pretraining, _run_identification, _run_attribution, _run_pattern_mining, _run_evaluation, _run_visualization, 
                   _model_name, _dataset_name, _old_process, _explanation_ratio=None, _min_iou=None, _included_categories=None, 
                   _filter_concepts=None, _max_concepts=None, _rule_methods=None, _min_support_params=None, _max_patterns_to_keep=None):
    """ Overrides selected important configs when called from outside without changing the source, e.g. from the Colab notebook """

    global run_pretraining
    global run_identification
    global run_attribution
    global run_pattern_mining
    global run_evaluation
    global run_visualization
    global model_name
    global dataset_name
    global dataset_pure_name
    global current_model_setting
    global current_dataset_setting
    global load_model_from_disk
    global base_model_url
    global num_classes
    global base_num_classes
    global target_layer
    global excluded_concepts
    global full_dataset_url
    global class_titles
    global classes
    global class_names
    global old_process
    global rule_methods
    global min_support_params
    global max_patterns_to_keep
    global explanation_ratio
    global filter_concepts
    global max_concepts
    global included_categories
    global min_iou

    run_pretraining = _run_pretraining
    run_identification = _run_identification
    run_attribution = _run_attribution
    run_pattern_mining = _run_pattern_mining
    run_evaluation = _run_evaluation
    run_visualization = _run_visualization

    model_name = _model_name
    dataset_name = _dataset_name
    dataset_pure_name = dataset_name.split('_')[0]

    current_model_setting = model_settings[model_name]
    target_layer = current_model_setting['target_layer']
    base_model_url = current_model_setting['url']

    current_dataset_setting = dataset_settings[dataset_name] 
    load_model_from_disk = current_dataset_setting['load_model']
    num_classes = current_dataset_setting['num_classes']
    base_num_classes = current_dataset_setting['base_num_classes']
    full_dataset_url = current_dataset_setting['url']
    excluded_concepts = current_dataset_setting['excluded_concepts'] if exclude_similar_concepts and ('excluded_concepts' in current_dataset_setting) else []

    class_titles = extract_class_titles(dataset_name, binning_classes)
    classes = list(class_titles.keys())
    class_names = list(class_titles.values())

    old_process = _old_process
    rule_methods = ['cart'] if old_process else ['cart', 'exp', 'ids']
    if _rule_methods:
        rule_methods = _rule_methods

    if _min_support_params:
        min_support_params = _min_support_params

    if _max_patterns_to_keep:
        max_patterns_to_keep = _max_patterns_to_keep
    
    if _filter_concepts != None:
        filter_concepts = _filter_concepts

    if _max_concepts:
        max_concepts = _max_concepts

    if _explanation_ratio:
        explanation_ratio = _explanation_ratio

    if _included_categories:
        included_categories = _included_categories

    if _min_iou:
        min_iou = _min_iou
