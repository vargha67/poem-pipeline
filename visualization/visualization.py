import configs
from pattern_mining import pattern_utils
import math, os, json
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from IPython.display import display



def load_patterns (concepts_file_path, exp_patterns_file_path, ids_patterns_file_path, cart_patterns_file_path):
    all_patterns_list = []
	
    if "exp" in configs.rule_methods: 
        exp_patterns = pd.read_csv(exp_patterns_file_path)
        exp_patterns['method'] = 'Exp'
        all_patterns_list.append(exp_patterns)
        
    if "ids" in configs.rule_methods:
        ids_patterns = pd.read_csv(ids_patterns_file_path)
        ids_patterns['method'] = 'IDS'
        all_patterns_list.append(ids_patterns)
        
    if "cart" in configs.rule_methods:
        cart_patterns = pd.read_csv(cart_patterns_file_path)
        cart_patterns['method'] = 'CART'
        all_patterns_list.append(cart_patterns)

    all_patterns = pd.concat(all_patterns_list, ignore_index=True)
    patterns_to_remove = []
    exp_patterns_count = 0
    
    for i,pattern in all_patterns.iterrows():
        if configs.remove_inactivated_patterns:
            # Removing patterns with "no" features, because they're not very accurate and useful: 
            to_remove = False
            for attr in list(pattern.index): 
                pattern_value = pattern[attr]
                if (attr not in configs.meta_cols) and (pattern_value == configs.low_value):
                    to_remove = True
                    break
                    
            if to_remove:
                print('Pattern {{{}}} to be removed because of having inactivated features!'.format(get_pattern_description(pattern)))
                patterns_to_remove.append(i)
                continue

    if len(patterns_to_remove) > 0:
        all_patterns.drop(patterns_to_remove, axis=0, inplace=True)

    all_patterns['support'] = all_patterns['support'].round(2)
    all_patterns['confidence'] = all_patterns['confidence'].round(2)
    all_patterns['accuracy'] = all_patterns['accuracy'].round(2)

    concept_cols = list(set(all_patterns.columns) - set(configs.meta_cols))
    all_patterns['score'] = all_patterns.apply(lambda p: compute_pattern_score(p, concept_cols), axis=1)

    group_cols = list(all_patterns.columns)
    group_cols.remove('method')
    all_patterns_grouped = all_patterns.groupby(group_cols, as_index=False)

    all_patterns = all_patterns_grouped.agg({'method': lambda p: ', '.join(p.unique())})

    all_patterns.sort_values(by=['score', 'confidence', 'support', 'accuracy', 'method'], ascending=False, inplace=True)
    all_patterns.reset_index(drop=True, inplace=True)
    all_patterns.insert(loc=0, column='index', value=(all_patterns.index + 1))
    all_patterns['score'] = all_patterns['score'].round(2)

    all_patterns = all_patterns.iloc[:configs.max_patterns]
    all_patterns = all_patterns.loc[:, (all_patterns != -1).any(axis=0)]
    concept_cols = list(set(all_patterns.columns) - set(configs.meta_cols))

    image_concepts = pd.read_csv(concepts_file_path)
    concepts_to_keep = set(concept_cols).union(set(['pred', 'label', 'id', 'file', 'path']))
    concepts_to_remove = list(set(image_concepts.columns) - concepts_to_keep)
    image_concepts.drop(concepts_to_remove, axis=1, inplace=True)

    return all_patterns, image_concepts, concept_cols



def get_pattern_description (pattern):
    antecedents = []
    for attr in list(pattern.index): 
        if (attr not in configs.meta_cols) and (pattern[attr] != -1):
            antecedents.append(attr + '=' + str(pattern[attr]))
    
    pred = pattern['pred']
    sup = pattern['support']
    conf = pattern['confidence']
    acc = pattern['accuracy']
    desc = 'If {}, then {} (sup: {}, conf: {}, acc: {})'.format(' & '.join(antecedents), configs.class_titles[pred], sup, conf, acc)
    return desc



def get_image_description (img_concepts, target_concept=None, target_channel=None):
    desc = 'Predicted ' + r'$\it{' + configs.class_titles[img_concepts['pred']] + '}$'
    desc += ', Labeled ' + r'$\it{' + configs.class_titles[img_concepts['label']] + '}$'

    if target_concept is None: 
        high_concepts = []
        mid_concepts = []
        for attr in list(img_concepts.index):
            if (attr not in ['pred', 'label', 'id', 'file', 'path']):
                if configs.binning_features:
                    if img_concepts[attr] == configs.mid_value:
                        mid_concepts.append(attr)
                    elif img_concepts[attr] == configs.high_value: 
                        high_concepts.append(attr)
                else:
                    if img_concepts[attr] == configs.high_value:
                        high_concepts.append(attr)
    
        if len(high_concepts) > 0:
            part_title = '\nHigh concepts: ' if configs.binning_features else '\nConcepts: '
            desc += part_title + r'$\it{' + ', '.join(high_concepts) + '}$'
        if len(mid_concepts) > 0:
            desc += '\nMid concepts: ' + r'$\it{' + ', '.join(mid_concepts) + '}$'
        if (len(high_concepts) == 0) and (len(mid_concepts) == 0):
            desc += '\nNo concepts activated'
    else:
        desc += '\nConcept ' + r'$\it{' + target_concept + '}$ highlighted'
        if target_channel != None:
            desc += ' (filter ' + r'$\it{' + str(target_channel) + '}$)'

    return desc



def list_image_activation_images (image_fname, activations_path, target_concept=None):
    ind = image_fname.rfind('.')
    image_fname_raw = image_fname[:ind]

    activations_info = {}
    for fname in os.listdir(activations_path):
        if not fname.startswith(image_fname_raw + "_"): 
            continue

        ind = fname.rfind('.')
        ext = fname[ind:]
        if (target_concept != None) and (not fname.endswith(target_concept + ext)):
            continue

        ind = len(image_fname_raw + "_")
        ind2 = fname.rfind(ext)
        main_body = fname[ind:ind2]
        parts = main_body.split('_')

        feature_value_cat = None
        channel = None
        concept = None
        if len(parts) == 2:
            channel = int(parts[0])
            concept = parts[1]
        elif len(parts) == 3:   # the case of binned features where the mid or high value of concept is also part of the file name
            feature_value_cat = parts[0]
            channel = int(parts[1])
            concept = parts[2]
        else:
            print('Error: Name of image file {} does not include feature value, concept and channel parts: {}'.format(fname, parts))
            continue

        if (target_concept != None) and (concept != target_concept):
            print('Error: Concept {} in name of file {} not matching the target concept {}'.format(concept, fname, target_concept))
            continue

        full_path = os.path.join(activations_path, fname)
        if concept in activations_info: 
            activations_info[concept].append((channel, full_path))
        else:
            activations_info[concept] = [(channel, full_path)]

    return activations_info



def prepare_image_items_for_display (matching_image_concepts, activations_path=None, target_concept=None):
    image_items = []
    for i, (ind, img_concepts) in enumerate(matching_image_concepts.iterrows()):
        img_item = {}
        img_path = img_concepts['path']
        img_fname = img_concepts['file']

        if target_concept is None:
            img_item['path'] = img_path
            img_item['desc'] = get_image_description(img_concepts)
            image_items.append(img_item)
        else:
            activations_info = list_image_activation_images(img_fname, activations_path, target_concept)
            
            if target_concept in activations_info:
                inf = activations_info[target_concept]
                item = inf[0]
                img_item['path'] = item[1]
                img_item['desc'] = get_image_description(img_concepts, target_concept, target_channel=item[0])
                image_items.append(img_item)
            else:
                print('Error: Activations info for image {} does not include the target concept {}: {}'.format(img_fname, target_concept, activations_info))

    return image_items



def plot_images (image_items, n_cols=3): 
    n_images = len(image_items)
    if n_images == 0:
        return

    n_rows = math.ceil(n_images / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axs = axs.flatten()
    for i, img_item in enumerate(image_items):
        ax = axs[i]
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        img_path = img_item['path']
        img_desc = img_item['desc']
        img = cv2.imread(img_path)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        title = ax.set_title(img_desc, fontsize=11)
        #title.set_y(1.05)
        #fig.subplots_adjust(top=0.8, bottom=0.8)

    plt.tight_layout()
    plt.show()



def display_images_matching_pattern (pattern, image_concepts, activations_path=None, target_concept=None, max_images=20):
    matching_indices = pattern_utils.find_images_matching_pattern(image_concepts, pattern)
    matching_image_concepts = image_concepts.iloc[matching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(matching_image_concepts, activations_path, target_concept)
    
    print('\nImages matching pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)



def display_images_supporting_pattern_not_matching (pattern, image_concepts, activations_path=None, target_concept=None, max_images=20):
    nonmatching_indices = pattern_utils.find_images_supporting_pattern_not_matching(image_concepts, pattern)
    nonmatching_image_concepts = image_concepts.iloc[nonmatching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(nonmatching_image_concepts, activations_path, target_concept)

    print('\nImages supporting but not matching pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)



def display_images_matching_pattern_wrong_predicted (pattern, image_concepts, activations_path=None, target_concept=None, max_images=20):
    wrong_indices = pattern_utils.find_images_matching_pattern_wrong_predicted(image_concepts, pattern)
    wrong_image_concepts = image_concepts.iloc[wrong_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(wrong_image_concepts, activations_path, target_concept)

    print('\nImages matching but predicted wrong for pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)



def display_image_activated_concepts (img_concepts, activations_path):
    img_id = img_concepts['id']
    img_fname = img_concepts['file']
    activations_info = list_image_activation_images(img_fname, activations_path)
    
    for con,inf in activations_info.items():
        image_items = []
        for item in inf:
            img_item = {}
            img_item['path'] = item[1]
            img_item['desc'] = get_image_description(img_concepts, target_concept=con, target_channel=item[0])
            image_items.append(img_item)

        print('\nImage {} with file name {}, with concept {} highlighted\n'.format(img_id, img_fname, con))
        plot_images(image_items, n_cols=3)



def display_images_having_concept (image_concepts, target_concept, activations_path=None, max_images=20):
    matching_indices = pattern_utils.find_images_having_concept(image_concepts, target_concept)
    matching_image_concepts = image_concepts.iloc[matching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(matching_image_concepts, activations_path, target_concept)
    
    print('\nImages having concept {}\n'.format(target_concept))
    plot_images(image_items, n_cols=4)



def display_single_images (image_concepts, target_concept=None, target_channel=None):
    image_items = []
    for i, (ind, img_concepts) in enumerate(image_concepts.iterrows()):
        img_item = {}
        img_item['path'] = img_concepts['path']
        img_item['desc'] = get_image_description(img_concepts, target_concept, target_channel)
        image_items.append(img_item)

    plot_images(image_items, n_cols=4)



def compute_pattern_score (row, concept_cols):
    sup = row['support']
    conf = row['confidence']
    size = 0

    for col in concept_cols:
        if row[col] != -1:
            size += 1

    score = (sup * (conf ** 2)) / (size ** 2)
    return score



def get_feature_value_desc (val):
    if val == -1:
        return ''

    if not configs.binning_features:
        return 'yes' if (val == configs.high_value) else 'no'

    if val == configs.high_value:
        return 'yes'
    elif val == configs.mid_value:
        return 'maybe'
    elif val == configs.low_value:
        return 'no'
    else:
        a = math.floor(val)
        b = math.ceil(val)
        if (a == configs.low_value) and (b == configs.mid_value):
            return 'no/maybe'
        elif (a == configs.mid_value) and (b == configs.high_value):
            return 'maybe/yes'
    
    return ''



def display_patterns (all_patterns, concept_cols):
    # Use a copy of the patterns dataframe for display:
    all_patterns_df = all_patterns.copy(deep=True)
    all_patterns_df.set_index('index', inplace=True)
    all_patterns_df['pred'] = all_patterns_df['pred'].apply(lambda p: configs.class_titles[p])

    # For test:
    # all_patterns_df['support'] = all_patterns_df['support'].apply(lambda p: p if p >= 0.03 else 0.03)

    renamed_cols = {}
    for con in concept_cols:
        all_patterns_df[con] = all_patterns_df[con].apply(lambda v: get_feature_value_desc(v))
        if '-' in con:
            i = con.rfind('-')
            renamed_cols[con] = con[:i]

    for col in configs.meta_cols:
        renamed_cols[col] = col.upper()
        if col == 'pred':
            renamed_cols[col] = 'PREDICTION'

    all_patterns_df.rename(columns=renamed_cols, inplace=True)
    all_patterns_df = all_patterns_df.rename_axis(None)

    display(all_patterns_df)



def evaluate_all_patterns (identification_result_path, concepts_file_path, patterns_base_path, evaluation_result_file_path):
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
        eval_item = {}
        exp_patterns_file_path = os.path.join(patterns_base_path, 'exp_patterns_' + str(sup) + '.csv')
        ids_patterns_file_path = os.path.join(patterns_base_path, 'ids_patterns_' + str(sup) + '.csv')
        cart_patterns_file_path = os.path.join(patterns_base_path, 'cart_patterns_' + str(sup) + '.csv')
        patterns, _image_concepts, concept_cols = \
            load_patterns(concepts_file_path, exp_patterns_file_path, ids_patterns_file_path, cart_patterns_file_path)

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

