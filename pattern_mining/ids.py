import configs
from pattern_mining.pattern_utils import KL_divergence, accuracy, compute_base_predictions, load_concepts_data
import numpy as np
import pandas as pd
import math, os, datetime, random, itertools
from apyori import apriori



class rule:
    # rule is of the form if A == a and B == b, then class_1
    # one of the member variables is itemset - a set of patterns {(A,a), (B,b)}
    # the other member variable is class_label (e.g., class_1)
    def __init__(self,feature_list,value_list,class_label):
        self.itemset = set()
        self.class_label = None
        self.cover = 0   # refers to support
        self.correct_cover = 0   # refers to confidence
        self.accurate_cover = 0   # refers to accuracy
        self.class_covers = None
        self.add_item(feature_list,value_list)
        self.set_class_label(class_label)
    
    def add_item(self,feature_list,value_list):
        if len(feature_list) != len(value_list):
            print("Some error in inputting feature value pairs")
            return
        for i in range(0,len(feature_list)):
            self.itemset.add((feature_list[i],value_list[i]))
        
    def all_predicates_same(self, r):
        return self.itemset == r.itemset
    
    def class_label_same(self,r):
        return self.class_label == r.class_label
            
    def set_class_label(self,label):
        self.class_label = label
        
    def get_length(self):
        return len(self.itemset)
    
    def get_cover(self, df):
        dfnew = df.copy()
        for pattern in self.itemset: 
            dfnew = dfnew[dfnew[pattern[0]] == pattern[1]]
        return list(dfnew.index.values)

    def get_correct_cover(self, df, Y):
        indexes_points_covered = self.get_cover(df) # indices of all points satisfying the rule
        Y_arr = pd.Series(Y)                    # make a series of all Y labels
        labels_covered_points = list(Y_arr[indexes_points_covered])   # get a list only of Y labels of the points covered
        correct_cover = []
        for ind in range(0,len(labels_covered_points)):
            if labels_covered_points[ind] == self.class_label:
                correct_cover.append(indexes_points_covered[ind])
        return correct_cover, indexes_points_covered
    
    def get_incorrect_cover(self, df, Y):
        correct_cover, full_cover = self.get_correct_cover(df, Y)
        return (sorted(list(set(full_cover) - set(correct_cover))))
        
    # new methods: 
    def compute_cover_counts(self, X, Y, Y_true): 
        indexes = self.get_cover(X)
        Y_list = Y.iloc[indexes].tolist()
        Y_true_list = Y_true.iloc[indexes].tolist()

        sup = len(indexes)
        conf = 0
        acc = 0
        class_covers = {}
        for i,y in enumerate(Y_list):
            if y in class_covers: 
                class_covers[y] = class_covers[y] + 1
            else:
                class_covers[y] = 1

            if y == self.class_label:
                conf += 1
                y_true = Y_true_list[i]
                if y == y_true:
                    acc += 1
        
        self.cover = sup
        self.correct_cover = conf / sup
        self.accurate_cover = acc / conf
        self.class_covers = {k:(v/sup) for k,v in class_covers.items()}
        
    def matches_data(self, data):
        for attr in self.itemset:
            if data[attr[0]] != attr[1]:
                return False
        return True



# This function basically takes a data frame and a support threshold and returns itemsets which satisfy the threshold
def run_apriori(df, support_thres):
    # the idea is to basically make a list of strings out of df and run apriori api on it 
    # return the frequent itemsets
    dataset = []
    for i in range(0,df.shape[0]):
        temp = []
        for col_name in df.columns:
            temp.append(col_name+"="+str(df[col_name][i]))
        dataset.append(temp)

    results = list(apriori(dataset, min_support=support_thres))
    
    list_itemsets = []
    for ele in results:
        temp = []
        for pred in ele.items:
            temp.append(pred)
        list_itemsets.append(temp)

    return list_itemsets



# This function converts a list of itemsets (stored as list of lists of strings) into rule objects
def createrules(freq_itemsets, labels_set):
    # create a list of rule objects from frequent itemsets 
    list_of_rules = []
    for one_itemset in freq_itemsets:
        feature_list = []
        value_list = []
        for pattern in one_itemset:
            fea_val = pattern.split("=")
            feature_list.append(fea_val[0])
            value_list.append(fea_val[1])
        for each_label in labels_set:
            temp_rule = rule(feature_list,value_list,each_label)
            list_of_rules.append(temp_rule)

    return list_of_rules



# compute the maximum length of any rule in the candidate rule set
def max_rule_length(list_rules):
    len_arr = []
    for r in list_rules:
        len_arr.append(r.get_length())
    return max(len_arr)



# compute the number of points which are covered both by r1 and r2 w.r.t. data frame df
def overlap(r1, r2, df):
    return sorted(list(set(r1.get_cover(df)).intersection(set(r2.get_cover(df)))))



# computes the objective value of a given solution set
def func_evaluation(soln_set, list_rules, df, Y, lambda_array):
    # evaluate the objective function based on rules in solution set 
    # soln set is a set of indexes which when used to index elements in list_rules point to the exact rules in the solution set
    # compute f1 through f7 and we assume there are 7 lambdas in lambda_array
    f = [] #stores values of f1 through f7; 
    
    # f0 term
    f0 = len(list_rules) - len(soln_set) # |S| - size(R)
    f.append(f0)
    
    # f1 term
    Lmax = max_rule_length(list_rules)
    sum_rule_length = 0.0
    for rule_index in soln_set:
        sum_rule_length += list_rules[rule_index].get_length()
    
    f1 = Lmax * len(list_rules) - sum_rule_length
    f.append(f1)
    
    # f2 term - intraclass overlap
    sum_overlap_intraclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label == list_rules[r2_index].class_label:
                sum_overlap_intraclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f2 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_intraclass
    f.append(f2)
    
    # f3 term - interclass overlap
    sum_overlap_interclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label != list_rules[r2_index].class_label:
                sum_overlap_interclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f3 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_interclass
    f.append(f3)
    
    # f4 term - coverage of all classes
    classes_covered = set() # set
    for index in soln_set:
        classes_covered.add(list_rules[index].class_label)
    f4 = len(classes_covered)
    f.append(f4)
    
    # f5 term - accuracy
    sum_incorrect_cover = 0.0
    for index in soln_set:
        sum_incorrect_cover += len(list_rules[index].get_incorrect_cover(df,Y))
    f5 = df.shape[0] * len(list_rules) - sum_incorrect_cover
    f.append(f5)
    
    #f6 term - cover correctly with at least one rule
    atleast_once_correctly_covered = set()
    for index in soln_set:
        correct_cover, full_cover = list_rules[index].get_correct_cover(df,Y)
        atleast_once_correctly_covered = atleast_once_correctly_covered.union(set(correct_cover))
    f6 = len(atleast_once_correctly_covered)
    f.append(f6)
    
    obj_val = 0.0
    for i in range(7):
        obj_val += f[i] * lambda_array[i]
    
    #print(f)
    return obj_val



# deterministic local search algorithm which returns a solution set as well as the corresponding objective value
def deterministic_local_search(list_rules, df, Y, lambda_array, epsilon):
    # step by step implementation of deterministic local search algorithm in the 
    # FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 4-5)
    t_start = datetime.datetime.now()

    #initialize soln_set
    soln_set = set()
    n = len(list_rules)
    
    # step 1: find out the element with maximum objective function value and initialize soln set with it
    each_obj_val = []
    for ind in range(len(list_rules)):
        each_obj_val.append(func_evaluation(set([ind]), list_rules, df, Y, lambda_array))
        
    best_element = np.argmax(each_obj_val)
    soln_set.add(best_element)
    print("Initial rule: " + str(best_element))
    S_func_val = each_obj_val[best_element]
    
    restart_step2 = False
    
    # step 2: if there exists an element which is good, add it to soln set and repeat
    while True:
        each_obj_val = []
        
        for ind in set(range(len(list_rules))) - soln_set:
            func_val = func_evaluation(soln_set.union(set([ind])), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.add(ind)
                print("Adding rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break

            t_end = datetime.datetime.now()
            if ((t_end - t_start).total_seconds() * 1000) > configs.ids_timeout:
                print('Deterministic search timed out in add loop, returning the current results ...')
                return soln_set, S_func_val
                
        print('Add loop finished!')
        if restart_step2:
            print('Restarting step 2 ...')
            restart_step2 = False
            continue
            
        for ind in soln_set:
            func_val = func_evaluation(soln_set - set([ind]), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.remove(ind)
                print("Removing rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break

            t_end = datetime.datetime.now()
            if ((t_end - t_start).total_seconds() * 1000) > configs.ids_timeout:
                print('Deterministic search timed out in remove loop, returning the current results ...')
                return soln_set, S_func_val
        
        print('Remove loop finished!')
        if restart_step2:
            print('Restarting step 2 ...')
            restart_step2 = False
            continue
        
        print('Evaluating s1 and s2 ...')
        # Evaluation of s2 which is a very large set can take a very long time, and is the main performance bottleneck: 
        s1 = func_evaluation(soln_set, list_rules, df, Y, lambda_array)
        s2 = 0   # func_evaluation(set(range(len(list_rules))) - soln_set, list_rules, df, Y, lambda_array)
        
        print(s1)
        print(s2)
        
        if s1 >= s2:
            return soln_set, s1
        else: 
            return set(range(len(list_rules))) - soln_set, s2



# Helper function for smooth_local_search routine: Samples a set of elements based on delta 
def sample_random_set(soln_set, delta, len_list_rules):
    all_rule_indexes = set(range(len_list_rules))
    return_set = set()
    
    # sample in-set elements with prob. (delta + 1)/2
    p = (delta + 1.0)/2
    for item in soln_set:
        random_val = np.random.uniform()
        if random_val <= p:
            return_set.add(item)
    
    # sample out-set elements with prob. (1 - delta)/2
    p_prime = (1.0 - delta)/2
    for item in (all_rule_indexes - soln_set):
        random_val = np.random.uniform()
        if random_val <= p_prime:
            return_set.add(item)
    
    return return_set



# Helper function for smooth_local_search routine: Computes estimated gain of adding an element to the solution set
def estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, error_threshold):
    #assumes rule_x_index is not in soln_set 
    Exp1_func_vals = []
    Exp2_func_vals = []
    
    while(True):
        # first expectation term (include x)
        for i in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            temp_soln_set.add(rule_x_index)
            Exp1_func_vals.append(func_evaluation(temp_soln_set, list_rules, df, Y, lambda_array))
        
        # second expectation term (exclude x)
        for j in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            if rule_x_index in temp_soln_set:
                temp_soln_set.remove(rule_x_index)
            Exp2_func_vals.append(func_evaluation(temp_soln_set, list_rules, df, Y, lambda_array))
    
        # compute standard error of mean difference
        variance_Exp1 = np.var(Exp1_func_vals, dtype=np.float64)
        variance_Exp2 = np.var(Exp2_func_vals, dtype=np.float64)
        std_err = math.sqrt(variance_Exp1/len(Exp1_func_vals) + variance_Exp2/len(Exp2_func_vals))
        print("Standard Error "+str(std_err))
        
        if std_err <= error_threshold:
            break
            
    return np.mean(Exp1_func_vals) - np.mean(Exp2_func_vals)



# Helper function for smooth_local_search routine: Computes the 'estimate' of optimal value using random search 
def compute_OPT(list_rules, df, Y, lambda_array):
    opt_set = set()
    for i in range(len(list_rules)):
        r_val = np.random.uniform()
        if r_val <= 0.5:
            opt_set.add(i)
    return func_evaluation(opt_set, list_rules, df, Y, lambda_array)



# smooth local search algorithm which returns a solution set
def smooth_local_search(list_rules, df, Y, lambda_array, delta, delta_prime):
    # step by step implementation of smooth local search algorithm in the 
    # FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 6)
    t_start = datetime.datetime.now()

    # step 1: set the value n and OPT; initialize soln_set to empty
    n = len(list_rules)
    OPT = compute_OPT(list_rules, df, Y, lambda_array)
    print("2/n*n OPT value is "+str(2.0/(n*n)*OPT))
    
    soln_set = set()
    
    restart_omega_computations = False
    
    while(True):
        # step 2 & 3: for each element estimate omega within certain error_threshold; if estimated omega > 2/n^2 * OPT, then add 
        # the corresponding rule to soln set and recompute omega estimates again
        omega_estimates = []
        for rule_x_index in range(n):
                
            print("Estimating omega for rule "+str(rule_x_index))
            omega_est = estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, 1.0/(n*n) * OPT)
            omega_estimates.append(omega_est)
            print("Omega estimate is "+str(omega_est))
            
            if rule_x_index in soln_set:
                continue
            
            if omega_est > 2.0/(n*n) * OPT:
                # add this element to solution set and recompute omegas
                soln_set.add(rule_x_index)
                restart_omega_computations = True
                print("-----------------------")
                print("Adding to the solution set rule "+str(rule_x_index))
                print("-----------------------")
                break    

            t_end = datetime.datetime.now()
            if ((t_end - t_start).total_seconds() * 1000) > configs.ids_timeout:
                print('Smooth search timed out, returning the current results ...')
                return sample_random_set(soln_set, delta_prime, n)
        
        if restart_omega_computations: 
            restart_omega_computations = False
            continue
            
        # reaching this point of code means there is nothing more to add to the solution set, but we can remove elements
        for rule_ind in soln_set:
            if omega_estimates[rule_ind] < -2.0/(n*n) * OPT:
                soln_set.remove(rule_ind)
                restart_omega_computations = True
                
                print("Removing from the solution set rule "+str(rule_ind))
                break

            t_end = datetime.datetime.now()
            if ((t_end - t_start).total_seconds() * 1000) > configs.ids_timeout:
                print('Smooth search timed out, returning the current results ...')
                return sample_random_set(soln_set, delta_prime, n)
                
        if restart_omega_computations: 
            restart_omega_computations = False
            continue
            
        # reaching here means there is no element to add or remove from the solution set
        return sample_random_set(soln_set, delta_prime, n)



def apply_smooth_search(list_of_rules, df, Y, lambda_array, delta1, delta_prime1, delta2, delta_prime2):
    s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, delta1, delta_prime1)
    s2 = smooth_local_search(list_of_rules, df, Y, lambda_array, delta2, delta_prime2)
    f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array)
    f2 = func_evaluation(s2, list_of_rules, df, Y, lambda_array)
    if f1 > f2:
        return s1, f1
    else:
        return s2, f2



def predict_single_data(rules, data, majority_class):
    rand_num = random.randrange(10)
    #if rand_num == 0: print('Predicting data example:', data.to_dict())
    labels = set()
    matching_rules = []
    
    best_label = None
    best_cover = 0.0
    best_rule = None
    for i,r in enumerate(rules):
        if r.matches_data(data):
            labels.add(r.class_label)
            matching_rules.append(r)
            
            if r.correct_cover > best_cover: 
                best_label = r.class_label
                best_cover = r.correct_cover
                best_rule = r
                
    if best_label is None: 
        return majority_class, None
    else: 
        return best_label, best_rule



def get_pattern_description (pattern, n_rows):
    antecedents = []
    for item in pattern.itemset: 
        antecedents.append(item[0] + '=' + str(item[1]))
    
    pred = pattern.class_label
    sup = pattern.cover / n_rows
    conf = pattern.correct_cover
    acc = pattern.accurate_cover
    desc = 'If {}, then {} (sup: {}, conf: {}, acc: {})'.format(' & '.join(antecedents), configs.class_titles[pred], sup, conf, acc)
    return desc



def evaluate_predictions(rules, X, Y_list, class_rates, base_labels):
    classes = list(class_rates.keys())
    
    majority_class = None
    max_rate = 0
    for k,v in class_rates.items():
        if v > max_rate: 
            max_rate = v
            majority_class = k
    print('Class {} is the majority class with rate {}'.format(majority_class, max_rate))
    
    true_labels = []
    pred_labels = []
    conf_labels = []
    preds = []
    for i,row in X.iterrows(): 
        p, rule = predict_single_data(rules, row, majority_class)
        class_covers = rule.class_covers if rule != None else class_rates
        
        conf_y = [class_covers[k] if k in class_covers else 0 for k in classes]
        conf_labels.append(conf_y)
        
        pred_y = [1 if k==p else 0 for k in classes]
        pred_labels.append(pred_y)
        preds.append(p)
        
        y = Y_list[i]
        true_y = [1 if k==y else 0 for k in classes]
        true_labels.append(true_y)
        
    print('True, prediction, and base labels:', list(zip(true_labels, pred_labels, conf_labels, base_labels))[:5])
        
    base_KL = KL_divergence(true_labels, base_labels)
    final_KL = KL_divergence(true_labels, conf_labels)
    info_gain = base_KL - final_KL
    ids_acc = accuracy(Y_list, preds)
    return base_KL, final_KL, info_gain, ids_acc



def run_process(X, Y, Y_true, class_rates, base_labels, search_params, n_rows):
    print("----------------------")
    print("Running {} search with search params {}".format(('smooth' if configs.ids_smooth_search else 'deterministic'), search_params))
    t1 = datetime.datetime.now()
    Y_list = Y.to_list()

    itemsets = run_apriori(X, search_params['min_support'])
    list_of_rules = createrules(itemsets, list(set(Y_list)))
    
    t2 = datetime.datetime.now()
    print('Time for creating the candidate rules:', t2-t1)
    print('Number of candidate rules generated:', len(list_of_rules))

    if configs.remove_inactivated_patterns:
        filtered_list_of_rules = []
        for i,r in enumerate(list_of_rules):
            inactivated_rule = False
            for attr in r.itemset: 
                if attr[1] == '0':
                    inactivated_rule = True
                    break

            if not inactivated_rule:
                filtered_list_of_rules.append(r)
        
        list_of_rules = filtered_list_of_rules
        print('Number of candidate rules reduced to {} after removing inactivated patterns.'.format(len(list_of_rules)))
        if len(list_of_rules) == 0: 
            return [], 0, 0
        
    soln_set = None  # [1752, 1, 19, 69]
    obj_val = None

    if soln_set is None:
        if configs.ids_smooth_search:
            soln_set, obj_val = apply_smooth_search(list_of_rules, X, Y_list, search_params['lambda_array'], 
                search_params['delta1'], search_params['delta_prime1'], search_params['delta2'], search_params['delta_prime2'])
        else:
            soln_set, obj_val = deterministic_local_search(list_of_rules, X, Y_list, search_params['lambda_array'], search_params['epsilon'])

    t3 = datetime.datetime.now()
    print('Time for local search:', t3-t2)
    print('Best solution set indices with evaluation criteria {}: {}'.format(obj_val, soln_set))

    solution_rules = [r for i,r in enumerate(list_of_rules) if i in soln_set]
    for r in solution_rules: 
        r.compute_cover_counts(X, Y, Y_true)
        #r.print_rule(n_rows)
        print(get_pattern_description(r, n_rows))
        
    base_KL, final_KL, info_gain, ids_acc = evaluate_predictions(solution_rules, X, Y_list, class_rates, base_labels)
    t4 = datetime.datetime.now()
    print('Time for evaluating predictions:', t4-t3)
    print('Total time of this execution:', t4-t1)
    print('Base KL:', base_KL)
    print('Final KL:', final_KL)
    print('Info gain:', info_gain)
    print('Accuracy:', ids_acc)
    
    return solution_rules, info_gain, ids_acc



def save_rules(solution_rules, feature_names, n_rows, output_path):
    rows_list = []
    for r in solution_rules:
        row = {c:-1 for c in feature_names}
        row['pred'] = r.class_label
        row['support'] = r.cover / n_rows
        row['confidence'] = r.correct_cover
        row['accuracy'] = r.accurate_cover
        for item in r.itemset:
            row[item[0]] = item[1]
        rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.to_csv(output_path, index=False)



def run_ids (concepts_file_path, output_base_path):
    t_start = datetime.datetime.now()
    X, Y, Y_true = load_concepts_data(concepts_file_path, 'pred', 'label', ['id', 'file', 'path'])

    n_rows = len(Y.index)
    feature_names = list(X.columns)
    class_counts = Y.value_counts().to_dict()
    class_rates = {k:(v/n_rows) for k,v in class_counts.items()}
    print('class_rates:', class_rates)

    base_labels = compute_base_predictions(X, class_rates)

    params_grid = configs.ids_params_grid
    param_keys = list(params_grid.keys())
    param_values = list(params_grid.values())
    param_combinations = list(itertools.product(*param_values))
    output_path_list = []
    results = []
    print('Parameter combinations:', param_combinations)

    for comb in param_combinations:
        params = {k:v for k,v in zip(param_keys, comb)}
        solution_rules, info_gain, ids_acc = run_process(X, Y, Y_true, class_rates, base_labels, params, n_rows)
        results.append({'params': params, 'rule_set_size': len(solution_rules), 'info_gain': info_gain})
        output_path = os.path.join(output_base_path, 'ids_patterns_' + str(params['min_support']) + '.csv')
        save_rules(solution_rules, feature_names, n_rows, output_path)
        output_path_list.append(output_path)

    t_end = datetime.datetime.now()
    print("----------------------")
    print('Total time:', t_end - t_start)
    print('Results:', results)

    return output_path_list
