import configs
import math
import random
import pandas as pd



def KL_divergence(p, q): 
    eps = 1e-15
    sum = 0
    for i in range(len(p)):  # iterating over each data example
        pi = p[i]
        qi = q[i]
        for j in range(len(pi)):  # iterating over each class probability 
            pij = pi[j] + eps
            qij = qi[j] + eps
            sum += (pij * math.log(pij / qij))
            
    return sum



def accuracy(p, q):
    true = 0
    for i in range(len(p)):
        pi = p[i]
        qi = q[i]
        if pi == qi:
            true += 1
                
    return true / len(p)



def compute_base_predictions(X, class_rates):
    classes = list(class_rates.keys())
    rates = list(class_rates.values())

    base_labels = []
    for i,row in X.iterrows(): 
        b = random.choices(classes, rates)[0]
        base_y = [class_rates[k] for k in classes]
        base_labels.append(base_y)
        
    return base_labels



def load_concepts_data(file_path, class_column, label_column, extra_columns=[], to_str=False):
    df = pd.read_csv(file_path)
    Y = df[class_column]
    Y_true = df[label_column]
    X = df.drop([class_column, label_column], axis=1)
    if len(extra_columns) > 0:
        X.drop(extra_columns, axis=1, inplace=True)
    if to_str:
        cols = X.columns
        X[cols] = X[cols].astype(str)
    return X, Y, Y_true



def find_images_supporting_pattern (image_concepts, pattern):
    df = image_concepts.copy()
    for attr in list(pattern.index): 
        pattern_value = pattern[attr]
        if (attr not in configs.meta_cols) and (pattern_value != -1):
            if (not configs.binning_features) or (pattern_value % 1 == 0): 
                df = df[df[attr] == pattern_value]
            else:
                # Handling the case of 0.5 or 1.5 values for a pattern feature: 
                a = math.floor(pattern_value)
                b = math.ceil(pattern_value)
                print('attr {} with value {}, floor {}, and ceil {}'.format(attr, pattern_value, a, b))
                df = df[(df[attr] == a) | (df[attr] == b)]

    supporting_indices = list(df.index.values)
    return supporting_indices



def find_images_matching_pattern (image_concepts, pattern, supporting_indices=None): 
    if supporting_indices is None:
        supporting_indices = find_images_supporting_pattern(image_concepts, pattern)
    pattern_label = pattern['pred']

    matching_indices = []
    supporting_labels = list(image_concepts.iloc[supporting_indices]['pred'])

    for i,label in enumerate(supporting_labels):
        if label == pattern_label:
            matching_indices.append(supporting_indices[i])

    return matching_indices



def find_images_supporting_pattern_not_matching (image_concepts, pattern, supporting_indices=None, matching_indices=None):
    if supporting_indices is None:
        supporting_indices = find_images_supporting_pattern(image_concepts, pattern)
    if matching_indices is None:
        matching_indices = find_images_matching_pattern(image_concepts, pattern, supporting_indices)
    
    nonmatching_indices = sorted(list(set(supporting_indices) - set(matching_indices)))
    return nonmatching_indices



def find_images_matching_pattern_wrong_predicted (image_concepts, pattern, matching_indices=None):
    if matching_indices is None:
        matching_indices = find_images_matching_pattern(image_concepts, pattern)

    wrong_indices = []
    matching_concepts = image_concepts.iloc[matching_indices][['pred', 'label']]

    for i,row in matching_concepts.iterrows():
        if row['pred'] != row['label']:
            wrong_indices.append(i)

    return wrong_indices



def find_images_having_concept (image_concepts, target_concept): 
    image_target_concepts = list(image_concepts[target_concept])
    matching_indices = []

    for i,val in enumerate(image_target_concepts):
        if val == 1:
            matching_indices.append(i)

    return matching_indices



def find_first_supported_pattern_for_image (img_concepts, patterns):
    for i,pattern in patterns.iterrows():
        is_match = True
        for attr in list(pattern.index):
            if (attr not in configs.meta_cols) and (pattern[attr] != -1):
                if img_concepts[attr] != pattern[attr]:
                    is_match = False
                    break
        
        if is_match:
            return i, pattern
    
    return -1, None



def compute_pattern_class_covers (pattern, image_concepts, preds, classes):
    indexes = find_images_supporting_pattern(image_concepts, pattern)
    preds_list = preds.iloc[indexes].tolist()

    sup = len(indexes)
    class_covers = {c:0 for c in classes}
    for i,y in enumerate(preds_list):
        class_covers[y] += 1
    
    return {k:(v/sup) for k,v in class_covers.items()}



def compute_patterns_average_measures (patterns):
    cnt = len(patterns.index)
    avg_size = 0
    avg_sup = 0
    avg_conf = 0
    avg_score = 0

    for i,pattern in patterns.iterrows():
        avg_sup += pattern['support']
        avg_conf += pattern['confidence']
        avg_score += pattern['score']

        for attr in list(pattern.index): 
            if (attr not in configs.meta_cols) and (pattern[attr] != -1):
                avg_size += 1

    avg_size = avg_size / cnt
    avg_sup = avg_sup / cnt
    avg_conf = avg_conf / cnt
    avg_score = avg_score / cnt
    return avg_size, avg_sup, avg_conf, avg_score



def compute_patterns_info_gain (patterns, image_concepts, preds):
    n_patterns = len(patterns.index)
    n_rows = len(preds.index)
    class_counts = preds.value_counts().to_dict()
    class_rates = {k:(v/n_rows) for k,v in class_counts.items()}
    classes = list(class_rates.keys())
    preds_list = preds.to_list()

    base_labels = compute_base_predictions(image_concepts, class_rates)

    patterns_class_covers = []
    for i,pattern in patterns.iterrows():
        class_covers = compute_pattern_class_covers(pattern, image_concepts, preds, classes)
        patterns_class_covers.append(class_covers)

    true_labels = []
    conf_labels = []
    for i,row in image_concepts.iterrows(): 
        ind, pattern = find_first_supported_pattern_for_image(row, patterns)
        class_covers = patterns_class_covers[ind] if ind != -1 else class_rates
        
        conf_y = [class_covers[k] if k in class_covers else 0 for k in classes]
        conf_labels.append(conf_y)
        
        y = preds_list[i]
        true_y = [1 if k==y else 0 for k in classes]
        true_labels.append(true_y)
        
    base_KL = KL_divergence(true_labels, base_labels)
    final_KL = KL_divergence(true_labels, conf_labels)
    info_gain = base_KL - final_KL
    avg_info_gain = info_gain / n_patterns
    return info_gain, avg_info_gain
