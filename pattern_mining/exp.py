import configs
from pattern_mining.pattern_utils import find_images_supporting_pattern, find_images_matching_pattern
import subprocess
import pandas as pd
import os



def compute_pattern_accuracies (image_concepts, patterns_file):
    exp_patterns = pd.read_csv(patterns_file)
    exp_patterns['accuracy'] = 0.0

    for i,pattern in exp_patterns.iterrows():
        pred = pattern['pred']
        conf = pattern['confidence']

        # Handling those patterns with lower than 0.5 confidence, which need to inverted to be useful (code below only works in case of not binning):
        if conf < 0.5:
            new_pred = pred
            new_conf = conf
            classes = list(configs.class_titles.keys())
            if len(classes) == 2:
                new_pred = 1 if pred == 0 else 0
                new_conf = 1.0 - conf
            else:
                pattern_cp = pattern.copy(deep=True)
                for c in classes:
                    if pred == c:
                        continue
                    pattern_cp['pred'] = c
                    supporting_indices = find_images_supporting_pattern(image_concepts, pattern_cp)
                    matching_indices = find_images_matching_pattern(image_concepts, pattern_cp, supporting_indices)
                    temp_conf = len(matching_indices) / len(supporting_indices)
                    if temp_conf > new_conf:
                        new_pred = c
                        new_conf = temp_conf
            
            exp_patterns.loc[i, 'pred'] = new_pred
            exp_patterns.loc[i, 'confidence'] = new_conf
            pattern['pred'] = new_pred
            pattern['confidence'] = new_conf
            print('Exp pattern with pred {} and conf {} changed to new pred {} and new conf {}'.format(pred, conf, new_pred, new_conf))

        pattern_label = pattern['pred']
        supporting_indices = find_images_supporting_pattern(image_concepts, pattern)
        matching_indices = find_images_matching_pattern(image_concepts, pattern, supporting_indices)
        matching_labels = list(image_concepts.iloc[matching_indices]['label'])

        accurate_indices = []
        for j,label in enumerate(matching_labels):
            if label == pattern_label:
                accurate_indices.append(matching_indices[j])

        conf = len(matching_indices)
        acc = len(accurate_indices) / conf
        exp_patterns.loc[i, 'accuracy'] = acc

    exp_patterns.to_csv(patterns_file, index=False)



def run_exp (concepts_file_path, output_base_path):
    image_concepts = pd.read_csv(concepts_file_path)
    concepts_meta_cols = ['pred', 'label', 'id', 'file', 'path']
    concept_cols = list(set(image_concepts.columns) - set(concepts_meta_cols))
    num_concepts = len(concept_cols)
    num_patterns = 30
    min_support_params = configs.exp_params_grid['min_support']
    remove_inactivated_patterns_num = 1 if configs.remove_inactivated_patterns else 0
    output_path_list = []

    for sup in min_support_params:
        output_path = os.path.join(output_base_path, 'exp_patterns_' + str(sup) + '.csv')
        print('Arguments to the program: {} {} {} {} {}'.format(configs.dataset_name, concepts_file_path, 
            num_concepts, num_patterns, remove_inactivated_patterns_num, output_path, sup))
        subprocess.run(["g++", "Explanations.cpp", "Lighthouse.cpp", "-o", "program"])
        subprocess.run(["./program", configs.dataset_name, concepts_file_path, num_concepts, num_patterns, 
            remove_inactivated_patterns_num, output_path, sup])
        output_path_list.append(output_path)

    for path in output_path_list:
        compute_pattern_accuracies(image_concepts, path)

    return output_path_list
