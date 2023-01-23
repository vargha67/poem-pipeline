import configs
from pattern_mining import pattern_utils
import os, json
import pandas as pd



def evaluate_all_patterns (concepts_file_path, patterns_base_path, evaluation_base_path):
    cases = ['cart', 'exp', 'ids', 'ensemble'] if not configs.old_process else ['cart']
    eval_results = {}

    for k in cases:
        eval_results[k] = {
            'total_avg_size': 0, 
            'total_avg_sup': 0,
            'total_avg_conf': 0,
            'total_avg_score': 0,
            'total_info_gain': 0,
            'total_avg_info_gain': 0,
            'pattern_measures': {sup:{} for sup in configs.min_support_params}
        }

    image_concepts, preds, labels = pattern_utils.load_concepts_data(concepts_file_path, 'pred', 'label', ['id', 'file', 'path'])

    for i, sup in enumerate(configs.min_support_params):
        exp_patterns_file_path = os.path.join(patterns_base_path, 'exp_patterns_' + str(sup) + '.csv')
        ids_patterns_file_path = os.path.join(patterns_base_path, 'ids_patterns_' + str(sup) + '.csv')
        cart_patterns_file_path = os.path.join(patterns_base_path, 'cart_patterns_' + str(sup) + '.csv')

        case_patterns = {}
        for k in cases:
            rule_methods = [k]
            if k == 'ensemble':
                rule_methods = ['cart', 'exp', 'ids']

            patterns, _, _ = pattern_utils.load_patterns(concepts_file_path, exp_patterns_file_path, 
                ids_patterns_file_path, cart_patterns_file_path, rule_methods)
            case_patterns[k] = patterns

        min_num_patterns = min([len(v.index) for k,v in case_patterns.items()])

        for k in cases:
            patterns = case_patterns[k].iloc[:min_num_patterns]

            avg_size, avg_sup, avg_conf, avg_score = pattern_utils.compute_patterns_average_measures(patterns)
            info_gain, avg_info_gain = pattern_utils.compute_patterns_info_gain(patterns, image_concepts, preds)

            eval_results[k]['pattern_measures'][sup]['n_patterns'] = len(patterns.index)
            eval_results[k]['pattern_measures'][sup]['avg_size'] = round(avg_size, 2)
            eval_results[k]['pattern_measures'][sup]['avg_sup'] = round(avg_sup, 2)
            eval_results[k]['pattern_measures'][sup]['avg_conf'] = round(avg_conf, 2)
            eval_results[k]['pattern_measures'][sup]['avg_score'] = round(avg_score, 2)
            eval_results[k]['pattern_measures'][sup]['info_gain'] = round(info_gain, 2)
            eval_results[k]['pattern_measures'][sup]['avg_info_gain'] = round(avg_info_gain, 2)

            eval_results[k]['total_avg_size'] += avg_size
            eval_results[k]['total_avg_sup'] += avg_sup
            eval_results[k]['total_avg_conf'] += avg_conf
            eval_results[k]['total_avg_score'] += avg_score
            eval_results[k]['total_info_gain'] += info_gain
            eval_results[k]['total_avg_info_gain'] += avg_info_gain

            concepts_set = set()
            for j,pattern in patterns.iterrows():
                for attr in list(pattern.index): 
                    if (attr not in configs.meta_cols) and (pattern[attr] != -1):
                        concepts_set.add(attr)

            eval_results[k]['pattern_measures'][sup]['concepts'] = list(concepts_set)

    cnt = len(configs.min_support_params)
    evaluations_path_list = []
    for k in cases:
        eval_results[k]['total_avg_size'] = round(eval_results[k]['total_avg_size'] / cnt, 2)
        eval_results[k]['total_avg_sup'] = round(eval_results[k]['total_avg_sup'] / cnt, 2)
        eval_results[k]['total_avg_conf'] = round(eval_results[k]['total_avg_conf'] / cnt, 2)
        eval_results[k]['total_avg_score'] = round(eval_results[k]['total_avg_score'] / cnt, 2)
        eval_results[k]['total_info_gain'] = round(eval_results[k]['total_info_gain'] / cnt, 2)
        eval_results[k]['total_avg_info_gain'] = round(eval_results[k]['total_avg_info_gain'] / cnt, 2)

        evaluation_file_path = os.path.join(evaluation_base_path, 'evaluation_' + k + '.json')
        with open(evaluation_file_path, 'w') as f:
            json.dump(eval_results[k], f, indent=4)
        evaluations_path_list.append(evaluation_file_path)

    return evaluations_path_list



def evaluate_all_patterns_old (identification_result_path, concepts_file_path, patterns_base_path, evaluation_result_file_path):
    eval_results = {}
    if os.path.exists(evaluation_result_file_path):
        with open(evaluation_result_file_path, 'r') as f:
            eval_results = json.load(f)

    concepts = None
    if configs.old_process:
        tally_path = os.path.join(identification_result_path, 'tally.csv')
        tally_data = pd.read_csv(tally_path)
        tally_data = tally_data[tally_data['score'] > configs.min_iou]

        concepts_list = tally_data['label'].tolist()
        concepts = list(set(concepts_list))
        concepts.sort()
    else:
        tally_path = os.path.join(identification_result_path, 'report.json')
        tally_data = {}
        with open(tally_path, 'r') as f:
            tally_data = json.load(f) 

        concepts_set = set()
        for ch_item in tally_data['units']:
            if ch_item['iou'] > configs.min_iou:
                concepts_set.add(ch_item['label'])

        concepts = list(concepts_set)
        concepts.sort()

    eval_results['concepts'] = concepts
    eval_results['num_concepts'] = len(concepts)

    image_concepts, preds, labels = pattern_utils.load_concepts_data(concepts_file_path, 'pred', 'label', ['id', 'file', 'path'])
    filtered_concepts = list(image_concepts.columns)
    eval_results['filtered_concepts'] = filtered_concepts
    eval_results['num_filtered_concepts'] = len(filtered_concepts)

    pattern_measures = []
    total_avg_size = 0
    total_avg_sup = 0
    total_avg_conf = 0
    total_avg_score = 0
    total_info_gain = 0
    total_avg_info_gain = 0
    cnt = len(configs.min_support_params)

    for sup in configs.min_support_params:
        eval_item = {
            'min_support': sup
        }
        exp_patterns_file_path = os.path.join(patterns_base_path, 'exp_patterns_' + str(sup) + '.csv')
        ids_patterns_file_path = os.path.join(patterns_base_path, 'ids_patterns_' + str(sup) + '.csv')
        cart_patterns_file_path = os.path.join(patterns_base_path, 'cart_patterns_' + str(sup) + '.csv')
        patterns, _image_concepts, concept_cols = \
            pattern_utils.load_patterns(concepts_file_path, exp_patterns_file_path, ids_patterns_file_path, cart_patterns_file_path)

        avg_size, avg_sup, avg_conf, avg_score = pattern_utils.compute_patterns_average_measures(patterns)
        eval_item['avg_size'] = round(avg_size, 2)
        eval_item['avg_sup'] = round(avg_sup, 2)
        eval_item['avg_conf'] = round(avg_conf, 2)
        eval_item['avg_score'] = round(avg_score, 2)
        total_avg_size += avg_size
        total_avg_sup += avg_sup
        total_avg_conf += avg_conf
        total_avg_score += avg_score

        info_gain, avg_info_gain = pattern_utils.compute_patterns_info_gain(patterns, image_concepts, preds)
        eval_item['info_gain'] = round(info_gain, 2)
        eval_item['avg_info_gain'] = round(avg_info_gain, 2)
        total_info_gain += info_gain
        total_avg_info_gain += avg_info_gain

        pattern_measures.append(eval_item)

    eval_results['total_avg_size'] = round(total_avg_size / cnt, 2)
    eval_results['total_avg_sup'] = round(total_avg_sup / cnt, 2)
    eval_results['total_avg_conf'] = round(total_avg_conf / cnt, 2)
    eval_results['total_avg_score'] = round(total_avg_score / cnt, 2)
    eval_results['total_info_gain'] = round(total_info_gain / cnt, 2)
    eval_results['total_avg_info_gain'] = round(total_avg_info_gain / cnt, 2)
    eval_results['pattern_measures'] = pattern_measures

    with open(evaluation_result_file_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
