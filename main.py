import os, sys, shutil, datetime

current_path = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(current_path, 'new_dissection'))
    sys.path.append(os.path.join(current_path, 'old_dissection'))
except Exception as ex:
    print('Exception in adding to sys path:', ex)

import gc
import gdown
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
    """ Main routine which runs the pipeline steps based on the configs """

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
    broden_dataset_file_path = broden_dataset_path + '.zip'

    full_dataset_path = os.path.join(datasets_path, configs.dataset_name)
    full_dataset_file_path = full_dataset_path + '.zip'

    base_model_file_path = os.path.join(models_path, configs.model_name + '_' + configs.dataset_pure_name + '.pth')
    segmenter_model_path = os.path.join(models_path, 'segmodel')
    segmenter_model_file_path = segmenter_model_path + '.zip'

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

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    t_start_total = datetime.datetime.now()

    # Model pretraining/finetuning: 
    if configs.run_pretraining:
        t_start = datetime.datetime.now()
        if not os.path.exists(full_dataset_path):
            gdown.download(url=configs.full_dataset_url, output=full_dataset_file_path, quiet=True, fuzzy=True)
            shutil.unpack_archive(full_dataset_file_path, datasets_path)

        if not os.path.exists(base_model_file_path):
            gdown.download(url=configs.base_model_url, output=base_model_file_path, quiet=True, fuzzy=True)

        pretraining.pretrain_model(full_dataset_path, dataset_path, base_model_file_path, model_file_path)
        gc.collect()
        torch.cuda.empty_cache()
        t_end = datetime.datetime.now()
        print('Time spent for pretraining:', t_end - t_start)

    # Concept identification: 
    if configs.run_identification:
        t_start = datetime.datetime.now()
        if configs.old_process: 
            if not os.path.exists(broden_dataset_path):
                gdown.download(url=configs.broden_dataset_url, output=broden_dataset_file_path, quiet=True, fuzzy=True)
                shutil.unpack_archive(broden_dataset_file_path, datasets_path)

            old_identification.concept_identification(broden_dataset_path, model_file_path, identification_result_path)
        else:
            if not os.path.exists(segmenter_model_path):
                gdown.download(url=configs.segmenter_model_url, output=segmenter_model_file_path, quiet=True, fuzzy=True)
                shutil.unpack_archive(segmenter_model_file_path, models_path)

            new_identification.concept_identification(dataset_path, model_file_path, segmenter_model_path, identification_result_path)

        utils.save_imported_packages(packages_file_path)
        gc.collect()
        torch.cuda.empty_cache()
        t_end = datetime.datetime.now()
        print('Time spent for identification:', t_end - t_start)

    # Concept attribution:
    if configs.run_attribution:
        t_start = datetime.datetime.now()
        if not os.path.exists(segmenter_model_path):
            gdown.download(url=configs.segmenter_model_url, output=segmenter_model_file_path, quiet=True, fuzzy=True)
            shutil.unpack_archive(segmenter_model_file_path, models_path)

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
        t_end = datetime.datetime.now()
        print('Time spent for attribution:', t_end - t_start)

    # Pattern mining using CART, IDS and Explanation Tables: 
    if configs.run_pattern_mining:
        t_start = datetime.datetime.now()
        if 'cart' in configs.rule_methods:
            patterns_path_list = cart.run_cart(concepts_file_path, cart_patterns_path)

        if 'ids' in configs.rule_methods:
            patterns_path_list = ids.run_ids(concepts_file_path, ids_patterns_path)

        if 'exp' in configs.rule_methods:
            patterns_path_list = exp.run_exp(concepts_file_path, exp_patterns_path)

        t_end = datetime.datetime.now()
        print('Time spent for pattern mining:', t_end - t_start)

    # Patterns evaluation: 
    if configs.run_evaluation:
        t_start = datetime.datetime.now()
        evaluations_path_list = evaluation.evaluate_all_patterns(concepts_file_path, cart_patterns_path, ids_patterns_path, 
                                                                 exp_patterns_path, evaluation_result_path)
        t_end = datetime.datetime.now()
        print('Time spent for evaluation:', t_end - t_start)

    # Patterns visualization:
    if configs.run_visualization:
        visualization.visualize_patterns(concepts_file_path, cart_patterns_path, ids_patterns_path, 
                                         exp_patterns_path, activation_images_path)

    t_end_total = datetime.datetime.now()
    print('Total pipeline time:', t_end_total - t_start_total)


if __name__ == '__main__':
    run_pipeline()
