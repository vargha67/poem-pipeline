import configs
from pattern_mining.pattern_utils import find_images_supporting_pattern, find_images_matching_pattern
import subprocess
import pandas as pd
import os, time, shutil



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



def run_exp (concepts_file_path, exp_patterns_path):
    if os.path.exists(exp_patterns_path):
        shutil.rmtree(exp_patterns_path)
    os.makedirs(exp_patterns_path)

    image_concepts = pd.read_csv(concepts_file_path)
    concepts_meta_cols = ['pred', 'label', 'id', 'file', 'path']
    concept_cols = list(set(image_concepts.columns) - set(concepts_meta_cols))
    num_concepts = len(concept_cols)
    num_patterns = 30
    remove_inactivated_patterns_num = 1 if configs.remove_inactivated_patterns else 0
    output_path_list = []

    current_path = os.path.abspath(os.path.dirname(__file__))
    explanations_path = os.path.join(current_path, 'exp', 'Explanations.cpp')
    lighthouse_path = os.path.join(current_path, 'exp', 'Lighthouse.cpp')
    res = subprocess.run(["g++", explanations_path, lighthouse_path, "-o", "program"], capture_output=True, universal_newlines=True)
    print('Compilation return code:', res.returncode)
    print('Compilation output:', res.stdout)
    print('Compilation error:', res.stderr)

    for sup in configs.min_support_params:
        output_path = os.path.join(exp_patterns_path, str(sup) + '.csv')
        print('Arguments to the program: {} {} {} {} {} {} {}'.format(configs.dataset_name, concepts_file_path, 
            num_concepts, num_patterns, remove_inactivated_patterns_num, output_path, sup))

        time.sleep(10)
        res = subprocess.run(["./program", configs.dataset_name, concepts_file_path, str(num_concepts), str(num_patterns), 
            str(remove_inactivated_patterns_num), output_path, str(sup)], capture_output=True, universal_newlines=True)
        print('Execution return code:', res.returncode)
        print('Execution output:', res.stdout)
        print('Execution error:', res.stderr)
        output_path_list.append(output_path)

    for path in output_path_list:
        compute_pattern_accuracies(image_concepts, path)

    return output_path_list
