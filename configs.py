import sys
from utils import extract_class_titles


# try:
#     sys.path.append('./new_dissection')
#     sys.path.append('./old_dissection')
# except Exception as ex:
#     print('Exception in adding to sys path:', ex)


model_settings = {
    "resnet18": {'target_layer': 'layer4.1.conv2'}, 
    "resnet50": {'target_layer': 'layer4.2.conv3'}, 
    "vgg16": {'target_layer': 'features.conv5_3'},   # 'features.28'
    "alexnet": {'target_layer': 'conv5'}
}

model_dataset_settings = {
    "resnet18_indoor_bedroom_kitchen": {'load_model': False, 'num_classes': 2, 'base_num_classes': 1000},
    "resnet18_places_bedroom_kitchen": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365},
    "resnet18_places_bedroom_kitchen_livingroom": {'load_model': True, 'num_classes': 3, 'base_num_classes': 365},
    "resnet18_places_coffeeshop_restaurant": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365},
    "resnet18_imagenet_minivan_pickup": {'load_model': False, 'num_classes': 2, 'base_num_classes': 1000, 'excluded_concepts': ['pickup', 'van']},   # ['car', 'bus', 'coach', 'truck', 'van']
    "resnet18_imagenet_laptop_mobile": {'load_model': False, 'num_classes': 2, 'base_num_classes': 1000, 'excluded_concepts': ['laptop', 'mobile', 'computer']},
    "resnet50_places_bedroom_kitchen": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365}, 
    "resnet50_places_bedroom_kitchen_livingroom": {'load_model': True, 'num_classes': 3, 'base_num_classes': 365},
    "resnet50_places_coffeeshop_restaurant": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365},
    "resnet50_imagenet_minivan_pickup": {'load_model': False, 'num_classes': 2, 'base_num_classes': 1000, 'excluded_concepts': ['pickup', 'van']},
    "vgg16_places_bedroom_kitchen": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365}, 
    "vgg16_places_coffeeshop_restaurant": {'load_model': True, 'num_classes': 2, 'base_num_classes': 365},
    "vgg16_places_bedroom_kitchen_livingroom": {'load_model': True, 'num_classes': 3, 'base_num_classes': 365},
    "vgg16_imagenet_minivan_pickup": {'load_model': False, 'num_classes': 2, 'base_num_classes': 1000, 'excluded_concepts': ['pickup', 'van']}
}


# Model and dataset settings: 
current_setting_title = "resnet18_places_bedroom_kitchen_livingroom"
title_parts = current_setting_title.split('_')
model_name = title_parts[0]
dataset_name = '_'.join(title_parts[1:])
dataset_pure_name = dataset_name.split('_')[0]

current_setting = model_dataset_settings[current_setting_title] 
load_model_from_disk = current_setting['load_model']   # Is true in case of having a model file pretrained on the target dataset (e.g. Places365) rather than the default dataset of Torchvision which is ImageNet
num_classes = current_setting['num_classes']
base_num_classes = current_setting['base_num_classes']
target_layer = model_settings[model_name]['target_layer']

# Pretraining settings: 
pretrain_mode = 'feature_extraction'   # full_fine_tuning, partial_fine_tuning, feature_extraction
target_classes = []
train_ratio = 0.7
train_batch_size = 32   # previously batch_size
epochs = 100
stop_patience = 10
cnn_dropout = 0.0
dense_dropout = 0.0
random_seed = 1

# Preprocessing settings: 
image_size = 224
norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)

# Concept identification settings:
old_process = False
seg_model_name = 'netpc'   # 'netpqc'
min_iou = 0.04
activation_high_thresh = 0.99
activation_low_thresh = 0.7
exclude_similar_concepts = True   # It is better to exclude a concept which is very similar to the dataset classes; e.g. laptop or computer concepts in laptop vs mobile dataset
excluded_concepts = current_setting['excluded_concepts'] if exclude_similar_concepts and ('excluded_concepts' in current_setting) else []

# Concept attribution settings:
batch_size = 16
gradient_high_thresh = 0.95   # previously activation_high_thresh
min_thresh_pixels_old = 1
min_thresh_pixels = 10
n_top_channels_per_concept = 3
overlay_opacity = 0.5
check_gradients = True
pool_gradients = False

# Segment-gradient overlap settings:
check_seg_overlap = True
overlap_mode = 'overlap_to_activation_ratio'   # 'overlap_pixels_count', 'overlap_to_union_ratio', 'overlap_to_activation_ratio', 'overlap_to_segmentation_ratio'
min_overlap_ratio = 0.5
min_overlap_pixels = 5
category_index_map = {
    'object': 0,
    'material': 1,
    'part': 2,
    'color': 3
}

# Concept filtering settings: 
filter_concepts_old = True
filter_concepts = True
low_variance_thresh = 0.99
max_concepts = 10

# Binning features settings:
binning_features = False
gradient_low_thresh = 0.7   # previously activation_low_thresh
high_value = 2 if binning_features else 1
mid_value = 1
low_value = 0

# Binning classes settings:
binning_classes = False
certainty_thresh = 0.6

# Pattern mining settings: 
remove_inactivated_patterns = False
class_titles = extract_class_titles(dataset_name, binning_classes)
classes = list(class_titles.keys())
class_names = list(class_titles.values())

min_support_params = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
ids_smooth_search = False
ids_timeout = 600000

# Pattern visualization settings:
meta_cols = ['index', 'pred', 'support', 'confidence', 'accuracy', 'method', 'score']
rule_methods = ['cart'] if old_process else ['cart', 'exp', 'ids']
ids_param = 0.01
cart_param = 0.03
exp_param = 0.03
max_patterns = 10

# Evaluation settings: 
max_patterns_to_keep = [5 for sup in min_support_params]



def update_configs(_current_setting_title, _old_process, _max_patterns_to_keep):
    global current_setting_title
    global title_parts
    global model_name
    global dataset_name
    global dataset_pure_name
    global current_setting
    global load_model_from_disk
    global num_classes
    global base_num_classes
    global target_layer
    global excluded_concepts
    global class_titles
    global classes
    global class_names
    global old_process
    global rule_methods
    global max_patterns_to_keep

    current_setting_title = _current_setting_title
    title_parts = current_setting_title.split('_')
    model_name = title_parts[0]
    dataset_name = '_'.join(title_parts[1:])
    dataset_pure_name = dataset_name.split('_')[0]

    current_setting = model_dataset_settings[current_setting_title] 
    load_model_from_disk = current_setting['load_model']
    num_classes = current_setting['num_classes']
    base_num_classes = current_setting['base_num_classes']
    target_layer = model_settings[model_name]['target_layer']
    excluded_concepts = current_setting['excluded_concepts'] if exclude_similar_concepts and ('excluded_concepts' in current_setting) else []

    class_titles = extract_class_titles(dataset_name, binning_classes)
    classes = list(class_titles.keys())
    class_names = list(class_titles.values())

    old_process = _old_process
    rule_methods = ['cart'] if old_process else ['cart', 'exp', 'ids']
    max_patterns_to_keep = _max_patterns_to_keep
