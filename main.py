import os, sys

current_path = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(current_path, 'new_dissection'))
    sys.path.append(os.path.join(current_path, 'old_dissection'))
except Exception as ex:
    print('Exception in adding to sys path:', ex)

import gc
import numpy as np
import torch
import configs, utils
from pretraining import pretraining
from identification import old_identification, new_identification
from attribution import old_attribution, new_attribution
from pattern_mining import cart, exp, ids
from evaluation import evaluation
from visualization import visualization


def run_pipeline():

    # Initial configurations: 
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda available:", torch.cuda.is_available())

    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(configs.random_seed)

    # Setting the main paths: 
    datasets_path = os.path.join(current_path, 'datasets')
    models_path = os.path.join(current_path, 'models')
    results_path = os.path.join(current_path, 'results')

    packages_file_path = os.path.join(results_path, 'packages.log')
    broden_dataset_path = os.path.join(datasets_path, 'broden1_224')
    full_dataset_path = os.path.join(datasets_path, configs.dataset_name)
    #dataset_path = os.path.join(datasets_path, 'dataset')

    base_model_file_path = os.path.join(models_path, configs.model_name + '_' + configs.dataset_pure_name + '.pth')
    segmenter_model_path = os.path.join(models_path, 'segmodel')
    #model_file_path = os.path.join(models_path, 'model.pth')

    pretraining_result_path = os.path.join(results_path, 'pretraining')
    identification_result_path = os.path.join(results_path, 'identification')
    attribution_result_path = os.path.join(results_path, 'attribution')
    cart_patterns_path = os.path.join(results_path, 'cart_patterns')
    ids_patterns_path = os.path.join(results_path, 'ids_patterns')
    exp_patterns_path = os.path.join(results_path, 'exp_patterns')
    evaluation_result_path = os.path.join(results_path, 'evaluation')

    dataset_path = os.path.join(pretraining_result_path, 'dataset')
    model_file_path = os.path.join(pretraining_result_path, 'model.pth')

    concepts_file_path = os.path.join(attribution_result_path, 'image_concepts.csv')
    channels_file_path = os.path.join(attribution_result_path, 'image_channels.csv')
    activation_images_path = os.path.join(attribution_result_path, 'activation_images')
    concepts_evaluation_file_path = os.path.join(attribution_result_path, 'evaluation_concepts.json')


    # Model pretraining: 
    if configs.run_pretraining:
        pretraining.pretrain_model(full_dataset_path, dataset_path, base_model_file_path, model_file_path)
        gc.collect()
        torch.cuda.empty_cache()

    # Concept identification: 
    if configs.run_identification:
        if configs.old_process: 
            old_identification.concept_identification(broden_dataset_path, model_file_path, identification_result_path)
        else:
            new_identification.concept_identification(dataset_path, model_file_path, segmenter_model_path, identification_result_path)

        utils.save_imported_packages(packages_file_path)
        gc.collect()
        torch.cuda.empty_cache()

    # Concept attribution:
    if configs.run_attribution:
        if configs.old_process:
            old_attribution.concept_attribution(dataset_path, model_file_path, segmenter_model_path, identification_result_path, 
                                                concepts_file_path, channels_file_path, activation_images_path,
                                                concepts_evaluation_file_path)
        else: 
            new_attribution.concept_attribution(dataset_path, model_file_path, segmenter_model_path, identification_result_path, 
                                                concepts_file_path, channels_file_path, activation_images_path, 
                                                concepts_evaluation_file_path)

        utils.save_imported_packages(packages_file_path)
        gc.collect()
        torch.cuda.empty_cache()

    # Pattern mining using CART, IDS and Explanation Tables: 
    if configs.run_pattern_mining:
        if 'cart' in configs.rule_methods:
            patterns_path_list = cart.run_cart(concepts_file_path, cart_patterns_path)

        if 'ids' in configs.rule_methods:
            patterns_path_list = ids.run_ids(concepts_file_path, ids_patterns_path)

        if 'exp' in configs.rule_methods:
            patterns_path_list = exp.run_exp(concepts_file_path, exp_patterns_path)

    # Patterns evaluation: 
    if configs.run_evaluation:
        evaluations_path_list = evaluation.evaluate_all_patterns(concepts_file_path, cart_patterns_path, ids_patterns_path, 
                                                                 exp_patterns_path, evaluation_result_path)

    # Patterns visualization:
    if configs.run_visualization:
        visualization.visualize_patterns(concepts_file_path, cart_patterns_path, ids_patterns_path, 
                                         exp_patterns_path, activation_images_path)


if __name__ == '__main__':
    run_pipeline()
