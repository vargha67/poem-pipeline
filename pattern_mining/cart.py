import configs
from pattern_mining.pattern_utils import KL_divergence, compute_base_predictions, load_concepts_data
import os, itertools, copy, datetime
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree



def count_tree_leaves(model):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    n_leaves = 0
    max_depth = 0
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        if depth > max_depth: 
            max_depth = depth

        is_split_node = children_left[node_id] != children_right[node_id]   # Different left and right children means split node
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            n_leaves += 1
            
    return n_nodes, n_leaves, max_depth



def get_rule_feature_value(thresh, is_left):
    if not configs.binning_features:
        return 0 if is_left else 1

    if is_left: 
        if thresh <= 0.5:
            return 0
        else:
            return 0.5   # means either 0 or 1
    else:
        if thresh > 1.0:
            return 2
        else:
            return 1.5   # means either 1 or 2



def get_pattern_description (pattern):
    antecedents = []
    for attr in list(pattern.keys()): 
        if (attr not in configs.meta_cols) and (pattern[attr] != -1):
            antecedents.append(attr + '=' + str(pattern[attr]))
    
    pred = pattern['pred']
    sup = pattern['support']
    conf = pattern['confidence']
    acc = pattern['accuracy']
    desc = 'If {}, then {} (sup: {}, conf: {}, acc: {})'.format(' & '.join(antecedents), configs.class_titles[pred], sup, conf, acc)
    return desc



def extract_save_rules(model, feature_names, leaves_stats, n_rows, output_path):
    rows_list = []
    node_features = model.tree_.feature
    node_thresholds = model.tree_.threshold
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    stack = [(0, [])]  # start with the root node id (0) and the empty path
    while len(stack) > 0:
        node_id, path = stack.pop()

        is_split_node = children_left[node_id] != children_right[node_id]   # Different left and right children means split node
        if is_split_node:
            left_path = copy.deepcopy(path)
            right_path = copy.deepcopy(path)
            thresh = node_thresholds[node_id]
            left_value = get_rule_feature_value(thresh, True)
            right_value = get_rule_feature_value(thresh, False)
            left_path.append((node_id, left_value))
            right_path.append((node_id, right_value))
            stack.append((children_left[node_id], left_path))
            stack.append((children_right[node_id], right_path))
        else:
            stat = leaves_stats[node_id]
            row = {}
            row = {c:-1 for c in feature_names}
            row['pred'] = stat['class']
            row['support'] = stat['support'] / n_rows
            row['confidence'] = stat['confidence']
            row['accuracy'] = stat['accuracy']
            for node_item in path:
                node, value = node_item
                feature_index = node_features[node]
                feature_name = feature_names[feature_index]
                row[feature_name] = value
                
            print(get_pattern_description(row))
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.to_csv(output_path, index=False)



def evaluate_predictions(Y_list, classes, base_labels, preds, pred_leaves, leaves_stats):
    true_count = 0
    true_labels = []
    pred_labels = []
    conf_labels = []
    for i,y in enumerate(Y_list):
        p = preds[i]
        p_leaf = pred_leaves[i]
        item = leaves_stats[p_leaf]
        class_covers = item['class_covers']
        
        if p == y:
            true_count += 1
            
        conf_y = [class_covers[k] if k in class_covers else 0 for k in classes]
        conf_labels.append(conf_y)
        
        pred_y = [1 if k==p else 0 for k in classes]
        pred_labels.append(pred_y)
        
        true_y = [1 if k==y else 0 for k in classes]
        true_labels.append(true_y)
        
    # print('True, prediction, and base labels:', list(zip(true_labels, pred_labels, conf_labels, base_labels, pred_leaves))[:5])
        
    base_KL = KL_divergence(true_labels, base_labels)
    final_KL = KL_divergence(true_labels, conf_labels)
    info_gain = base_KL - final_KL
    tree_acc = true_count / len(Y_list)
    return base_KL, final_KL, info_gain, tree_acc



def train_evaluate_tree(X, Y, Y_true, classes, base_labels, model_params, feature_names):
    print("----------------------")
    print('Training and evaluating tree with params:', model_params)
    t1 = datetime.datetime.now()
    Y_list = Y.to_list()
    Y_true_list = Y_true.to_list()
    
    model = tree.DecisionTreeClassifier(**model_params)
    model.fit(X, Y)
    
    preds = model.predict(X)
    pred_leaves = model.apply(X)
    
    leaves_stats = {}
    for i,n in enumerate(pred_leaves):
        y = Y_list[i]
        y_true = Y_true_list[i]
        p = preds[i]
        c = 1 if y == p else 0
        a = 1 if ((c == 1) and (y == y_true)) else 0
        if n in leaves_stats: 
            item = leaves_stats[n]
            item['support'] += 1
            item['confidence'] += c
            item['accuracy'] += a
            if item['class'] != p: 
                print('Different prediction {} with class {} found for leaf {}!'.format(p, item['class'], n))
            
            class_covers = item['class_covers']
            if y in class_covers:
                class_covers[y] = class_covers[y] + 1
            else:
                class_covers[y] = 1
        else: 
            class_covers = {}
            class_covers[y] = 1
            leaves_stats[n] = { 'class': p, 'support': 1, 'confidence': c, 'accuracy': a, 'class_covers': class_covers }
            
    for n,item in leaves_stats.items():
        sup = item['support']
        conf = item['confidence']
        acc = item['accuracy']
        item['confidence'] = conf / sup
        item['accuracy'] = acc / conf
        class_covers = item['class_covers']
        leaves_stats[n]['class_covers'] = {k:(v/sup) for k,v in class_covers.items()}
    # print('Leaves stats:', leaves_stats)
    
    n_nodes, n_leaves, max_depth = count_tree_leaves(model)
    print('Tree has {} nodes, {} leaves, and max depth of {}'.format(n_nodes, n_leaves, max_depth))
    
    base_KL, final_KL, info_gain, tree_acc = evaluate_predictions(Y_list, classes, base_labels, preds, pred_leaves, leaves_stats)
    
    t2 = datetime.datetime.now()
    print('Base KL:', base_KL)
    print('Final KL:', final_KL)
    print('Info gain:', info_gain)
    print('Accuracy:', tree_acc)
    print('Time for training and evaluating the tree:', t2-t1)
    
    tree_txt = tree.export_text(model, feature_names=feature_names, show_weights=True)
    # print(tree_txt)
    
    #fig = plt.figure(figsize=(25,20))
    #tree.plot_tree(model, feature_names=feature_names, class_names=configs.class_names, filled=True)
    #plt.show()
    #fig.savefig("tree.png")
    
    #tree.export_graphviz(model, "tree", feature_names=feature_names, class_names=configs.class_names)
    
    return model, leaves_stats, n_leaves, info_gain, tree_acc



def run_cart (concepts_file_path, output_base_path):
    t_start = datetime.datetime.now()
    X, Y, Y_true = load_concepts_data(concepts_file_path, 'pred', 'label', ['id', 'file', 'path'])   # Y is CNN model predictions, while Y_true is ground truth labels

    feature_names = list(X.columns)
    n_rows = len(Y.index)
    class_counts = Y.value_counts().to_dict()
    class_rates = {k:(v/n_rows) for k,v in class_counts.items()}
    classes = list(class_rates.keys())
    num_classes = len(classes)
    print('class_rates:', class_rates)

    base_labels = compute_base_predictions(X, class_rates)

    params_grid = {
        'criterion': ['entropy'], 
        'min_samples_leaf': configs.min_support_params
    }
    param_keys = list(params_grid.keys())
    param_values = list(params_grid.values())
    param_combinations = list(itertools.product(*param_values))

    output_path_list = []
    results = []
    # print('Parameter combinations:', param_combinations)

    for comb in param_combinations:
        params = {k:v for k,v in zip(param_keys, comb)}
        model, leaves_stats, n_leaves, info_gain, tree_acc = \
            train_evaluate_tree(X, Y, Y_true, classes, base_labels, params, feature_names)
        results.append({'params': params, 'leaves': n_leaves, 'info_gain': info_gain})
        output_path = os.path.join(output_base_path, 'cart_patterns_' + str(params['min_samples_leaf']) + '.csv')
        extract_save_rules(model, feature_names, leaves_stats, n_rows, output_path)
        output_path_list.append(output_path)
        
    t_end = datetime.datetime.now()
    print("----------------------")
    print('Total time:', t_end - t_start)
    print('Results:', results)

    return output_path_list
