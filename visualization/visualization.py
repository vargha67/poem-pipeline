import configs
from pattern_mining import pattern_utils
import math, os, json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from IPython.display import display


def get_pattern_description (pattern):
    """ Returns a human-readable representation of a rule/pattern """

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
    """ Returns the description below an image in the visualizations """

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
    """ Collects the paths of activation images for an image """

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
    """ Prepares the path and description of images for display """

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
    """ Plots the images in appropriate matrix format """

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
    """ Displays images matching a pattern's concepts and outcome """

    matching_indices = pattern_utils.find_images_matching_pattern(image_concepts, pattern)
    matching_image_concepts = image_concepts.iloc[matching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(matching_image_concepts, activations_path, target_concept)
    
    print('\nImages matching pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)


def display_images_supporting_pattern_not_matching (pattern, image_concepts, activations_path=None, target_concept=None, max_images=20):
    """ Displays images supporting a pattern's concepts, but not matching its outcome """

    nonmatching_indices = pattern_utils.find_images_supporting_pattern_not_matching(image_concepts, pattern)
    nonmatching_image_concepts = image_concepts.iloc[nonmatching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(nonmatching_image_concepts, activations_path, target_concept)

    print('\nImages supporting but not matching pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)


def display_images_matching_pattern_wrong_predicted (pattern, image_concepts, activations_path=None, target_concept=None, max_images=20):
    """ Displays images matching a pattern's concepts and outcome, but predicted incorrectly by the model """

    wrong_indices = pattern_utils.find_images_matching_pattern_wrong_predicted(image_concepts, pattern)
    wrong_image_concepts = image_concepts.iloc[wrong_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(wrong_image_concepts, activations_path, target_concept)

    print('\nImages matching but predicted wrong for pattern {{{}}}'.format(get_pattern_description(pattern)) + 
          (', with concept {} highlighted\n'.format(target_concept) if target_concept != None else '\n'))
    plot_images(image_items, n_cols=4)


def display_image_activated_concepts (img_concepts, activations_path):
    """ Displays the set of activation images for an image """

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
    """ Displays all images attributed to a concept """

    matching_indices = pattern_utils.find_images_having_concept(image_concepts, target_concept)
    matching_image_concepts = image_concepts.iloc[matching_indices].iloc[:max_images]
    image_items = prepare_image_items_for_display(matching_image_concepts, activations_path, target_concept)
    
    print('\nImages having concept {}\n'.format(target_concept))
    plot_images(image_items, n_cols=4)


def display_single_images (image_concepts, target_concept=None, target_channel=None):
    """ Displays a set of images in normal or activation forms """

    image_items = []
    for i, (ind, img_concepts) in enumerate(image_concepts.iterrows()):
        img_item = {}
        img_item['path'] = img_concepts['path']
        img_item['desc'] = get_image_description(img_concepts, target_concept, target_channel)
        image_items.append(img_item)

    plot_images(image_items, n_cols=4)


def get_feature_value_desc (val):
    """ Returns the appropriate label for a feature value """

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
    """ Displays the combined list of patterns from different methods """

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


def visualize_patterns (concepts_file_path, cart_patterns_path, ids_patterns_path, exp_patterns_path, 
                        activation_images_path, min_support=0.01, pattern_index=1, target_concept=None):
    """ Main process of visualizing the list of patterns and sample related images of a sample pattern from the list """

    print('----------------------------------------------')
    print('Pattern visualization ...')

    mpl.rcParams['lines.linewidth'] = 0.25
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 0.25
    pd.options.display.max_columns = None

    cart_patterns_file_path = os.path.join(cart_patterns_path, str(min_support) + '.csv')
    ids_patterns_file_path = os.path.join(ids_patterns_path, str(min_support) + '.csv')
    exp_patterns_file_path = os.path.join(exp_patterns_path, str(min_support) + '.csv')

    all_patterns, image_concepts, concept_cols = \
        pattern_utils.load_patterns(concepts_file_path, exp_patterns_file_path, ids_patterns_file_path, cart_patterns_file_path)

    display_patterns(all_patterns, concept_cols)

    pattern = all_patterns.loc[all_patterns['index'] == pattern_index].iloc[0]
    display_images_matching_pattern(pattern, image_concepts, activation_images_path, target_concept)

    display_images_supporting_pattern_not_matching(pattern, image_concepts, activation_images_path, target_concept)

    display_images_matching_pattern_wrong_predicted(pattern, image_concepts, activation_images_path, target_concept)
