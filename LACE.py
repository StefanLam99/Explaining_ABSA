from Counterfactuals import *
import time

def evaluation_main():
    model = 'Maria'
    if model == 'Olaf':
        write_path = 'data/LACE/LACE_predDifference' + model + str(2016)
        # write_path = 'data/LACE/LACE_fidelityComputationsPOS' + model + str(2016)
        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
    elif model == "Maria":
        write_path = 'data/LACE/LACE_predDifference' + model + str(2016)
        # write_path = 'data/LACE/LACE_fidelityComputationsPOS' + model + str(2016)
        input_file = 'data/programGeneratedData/768remainingtestdata2016.txt'
    B = classifier(model)
    size, polarity = getStatsFromFile(input_file)  # polarity is a vector with the classifications of the instances
    predictions = np.array([])
    probabilities = np.zeros((int(size), 3))

    begin = time.time()
    with open(write_path + '.txt', 'w') as results:
        for test_index in range(152, 153):
            print("Instance: ", test_index)
            x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = B.get_instance(test_index)
            pred, prob = B.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
            predictions = np.append(predictions, pred)
            print('Prediction of instance x: ', pred)
            probabilities[test_index, :] = prob
            print('Probabilities of instance x: ', prob)
            # print("Before omission: (left and right)", x_left, x_right)
            s = B.get_String_Sentence(B.x_left[test_index])
            t = B.get_String_Sentence(B.x_right[test_index])
            sentence = s + t
            print("Left and right sentence part: ", s, t)
            anchor_paths = [['very', 'creme', 'was', 'the', 'but', 'brulee', ',', 'and']]
            #['positive', 'nice', 'the', 'is'], ['we', 'there', 'the', 'fantastic', 'waiting', 'is', 'has', 'been', ',',
                                           #     'and']


            counterfactual = [['not weird', 'not odd', 'not strange', 'not obvious', 'delicious'], ['not', 'delicious', 'another'], ['weird', 'delicious', 'savory', 'not very', 'not all'], ['not weird', 'odd', 'delicious', 'very', 'concealer']]
                #[['not bad', 'not weird', 'not any', 'not little', 'not crazy'],['bad', 'not guy', 'not lot', 'never', 'not location'], ['bad', 'not guy', 'not lot', 'not never',
                 #                                                    'not location'], ['not bad', 'weird', 'not guy',
                 #                                                                      'not once', 'not even'], [
                #'not bad', 'not weird', 'any', 'not', 'all'], ['not bad', 'not weird', 'any', 'not not', 'situation'], [
                #'not bad', 'not weird', 'not any', 'little', 'shame']]
            #['not decent', 'not intriguing', 'not glorious', 'not unique', "'ve"], ['decent', 'dual']
            temp_subsets = get_subsets(counterfactual, sentence)
            print("Paths before formatting: ", counterfactual)
            print("Paths after formatting: ", temp_subsets)

            path = ['thing']
            x_left_omitted, x_left_omitted_len, x_right_omitted, x_right_omitted_len, y_true, target_word, target_len, dec_rule_right, dec_rule_right_len, dec_rule_left, dec_rule_left_len = \
                omit_subset(path, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, B)
            pred_diff, pred_neighbor = get_pred_difference(pred, prob, x_left_omitted, x_left_omitted_len, x_right_omitted,
                                                           x_right_omitted_len, y_true, target_word, target_len, B)
            print("These are prediction difference vectors for instance", test_index)
            print("Word", path, pred_diff)





def main():
    model = 'Maria'
    if model == 'Olaf':
        write_path = 'data/LACE/LACE_fidelityComputationsTEST' + model + str(2016)
        #write_path = 'data/LACE/LACE_fidelityComputationsPOS' + model + str(2016)
        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
    elif model == "Maria":
        write_path = 'data/LACE/LACE_fidelityComputationsTEST' + model + str(2016)
        #write_path = 'data/LACE/LACE_fidelityComputationsPOS' + model + str(2016)
        input_file = 'data/programGeneratedData/768remainingtestdata2016.txt'

    B = classifier(model)
    size, polarity = getStatsFromFile(input_file)  # polarity is a vector with the classifications of the instances
    predictions = np.array([])
    probabilities = np.zeros((int(size), 3))

    nlp = en_core_web_lg.load()
    neighbors = Neighbors(nlp)

    important_words = np.zeros((int(size), 3))
    instance_pred_diff = np.zeros((int(size), 3))
    #fidelity_scores = np.array([])
    fidelity_scores = np.zeros((int(size), 1))
    fidelity_chosen_rules = np.zeros((int(size), 1))
    begin = time.time()
    with open(write_path + '.txt', 'w') as results:
        for test_index in range(int(size)):
            r = check_random_state(2020)
            num_samples = 5000
            batch_size = 200
            # range(149, 150) range(255, 256) + range(218, 219) + range(260, 261))
            print("Instance: ", test_index)
            x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = B.get_instance(test_index)
            pred, prob = B.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
            predictions = np.append(predictions, pred)
            print('Prediction of instance x: ', pred)
            probabilities[test_index, :] = prob
            print('Probabilities of instance x: ', prob)
           # print("Before omission: (left and right)", x_left, x_right)
            s = B.get_String_Sentence(B.x_left[test_index])
            t = B.get_String_Sentence(B.x_right[test_index])
            sentence = s + t
            print("Left and right sentence part: ", s, t)
            print("Now we draw the tree: ")
            # DRAWING TREE
            pred_f, true_label, pred_c, sentence_matrix, set_features = data_POS(B, num_samples, test_index,neighbors)

            root_full = build_tree(sentence_matrix, set_features, 0)
            tree_full = Tree(root_full)
            #print_tree(root_full)
            root_leaf_paths = tree_full.get_paths()
            paths, path_labels = get_tree_paths(root_leaf_paths)

            print("Paths before formatting: ", paths)
            temp_subsets = get_subsets(paths, sentence)
            print("Paths after formatting: ", temp_subsets)
            n_subsets = len(temp_subsets)
            neighbor_probs = np.zeros((int(n_subsets), 3))
            neighbor_predictions = np.array([])
            rule_predictions = np.array([])
            print("We go over this many iterations:", n_subsets)
            for relevant_subset in range(n_subsets):
                #print("ITERATION", relevant_subset)
                subset = temp_subsets[relevant_subset]
                #print("Before left: ", x_left)
                #print("Before right: ", x_right)
                #print(len(x_right))
                #print(x_right.shape)
                #print("Subset: ", temp_subsets[relevant_subset])
                x_left_omitted, x_left_omitted_len, x_right_omitted, x_right_omitted_len, y_true, target_word, target_len, dec_rule_right, dec_rule_right_len, dec_rule_left, dec_rule_left_len = \
                    omit_subset(subset, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, B)
                pred_rule, prob_rule = B.get_prob(dec_rule_left, dec_rule_left_len, dec_rule_right, dec_rule_right_len, y_true, target_word, target_words_len)
                rule_predictions = np.append(rule_predictions, pred_rule)
                #print("After omission: (left and right)", x_left_omitted_len, x_right_omitted_len)
                #print("Test: ", dec_rule_right)
                #print("After: x_left_omitted", x_left_omitted)
                #print("After: x_right_omitted ", x_right_omitted)
                pred_diff, pred_neighbor = get_pred_difference(pred, prob, x_left_omitted, x_left_omitted_len, x_right_omitted,
                                                x_right_omitted_len, y_true, target_word, target_len, B)
                neighbor_probs[relevant_subset, :] = pred_diff
                neighbor_predictions = np.append(neighbor_predictions, pred_neighbor)
            #print("These are prediction difference vectors for instance", test_index)
            #print(neighbor_probs)
            #print("prediction differences for sentiment -1?", neighbor_probs[:, 0])
            #stats = np.max(neighbor_probs, 0)
            #print("Maximum values of all columns ", stats)

            #print(len(neighbor_probs))
            n_sentiments = 3
            fidelity_instance = get_instance_fid(rule_predictions, path_labels)
            chosen_rule, fidelity_rule = relevance_subsets(neighbor_probs, n_sentiments, temp_subsets, pred, path_labels, rule_predictions)
            for word in chosen_rule:

                x_left_omitted_word, x_left_omitted_word_len, x_right_omitted_word, x_right_omitted_word_len, y_true, target_word, target_len, word_rule_right, word_rule_right_len, word_rule_left, word_rule_left_len = \
                    omit_subset(word, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len,
                                B)
                #print(x_right_omitted_word)
                pred_diff_word, prob_diff_word = get_pred_difference(pred, prob, x_left_omitted_word,
                                                                     x_left_omitted_word_len, x_right_omitted_word, x_right_omitted_word_len, y_true, target_word, target_len, B)
                print('This word provides these prob differences: ', word, pred_diff_word, prob_diff_word)
            print("Fidelity of all rules of instance x is: ", fidelity_instance)
            print("We choose this rule: ", chosen_rule, "And this corresponding fidelity: ", fidelity_rule)
            print("Iterations done.")
            #print('Predictions for neighbors', neighbor_predictions)
            print('Rule predictions ', rule_predictions)
            print("Labels by decision tree: ", path_labels)
            #local_fidelity = get_tree_fidelity(rule_predictions, path_labels)
            #print("Fidelity for this instance is ", local_fidelity)
            #fidelity_scores = np.append(fidelity_scores, local_fidelity)
            fidelity_scores[test_index, :] = fidelity_instance
            fidelity_chosen_rules[test_index, :] = fidelity_rule
            #print(fidelity_scores)
            average_fidelity = np.average(fidelity_scores)
            average_rule_fid = np.average(fidelity_chosen_rules)
            print("Fidelity so far :", average_fidelity)
            print("Fidelity of chosen rules : ", average_rule_fid)
            #print("Standard deviation:", np.std(fidelity_scores))
            print("_________________________________________________")
            print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")

            results.write('Instance: ' + str(test_index) + '\n')
            results.write('Sentence of instance: ' + str(sentence) + '\n')
            results.write('Potentially relevant subsets: ' + str(temp_subsets) + '\n' )
            results.write('Amount of potentially relevant subsets: ' + str(n_subsets) + '\n')
            results.write('\n')
            results.write('Prediction for instance x by b(x): ' + str(pred) + '\n')
            results.write('Prediction label by decision tree: ' + str(path_labels) + '\n')
            results.write('Prediction of decision rules by b(.): ' + str(rule_predictions) + '\n')
            #results.write('Prediction of perturbed instances by b(z): ' + str(neighbor_predictions) + '\n')
            results.write('Most relevant subset (rule): ' + str(chosen_rule) + '\n')
            results.write('Fidelity for this chosen rule of instance: ' + str(fidelity_rule) + '\n')
            results.write('Fidelity for all rules of instance: ' + str(fidelity_instance) + '\n')
            results.write('\n')
            results.write('Average fidelity of chosen rules so far: ' + str(average_rule_fid) + '\n')
            results.write('Average fidelity of all rules so far: ' + str(average_fidelity) + '\n')
            results.write('Standard deviation of fidelity so far: ' + str(np.std(fidelity_scores)) + '\n')
            results.write('_________________________________________________________________' + '\n')

    results.close()
    # Compute running time
    end = time.time()
    seconds = end - begin
    print('It took: ' + str(seconds) + ' seconds')


def get_tree_paths(root_leaf_paths):
    # make tree from instance x
    paths = []
    sentiment = []
    keys = np.array([])

    for key in root_leaf_paths.keys():
        sentiment.append(key)
        paths += root_leaf_paths[key]
        for path in range(len(root_leaf_paths[key])):
            keys = np.append(keys, int(key))
    return paths, keys


def get_subsets(paths, sentence):
    """
    :param sentence:
    :param paths: root-leaf paths
    :param classifier_b: black box B
    :param car_instance: CAR
    :return:
    """
    formatted_subsets = []
    n_subsets = len(paths)

    for subset in range(n_subsets):
        temporary = []
        n_words = len(paths[subset])

        for word in range(n_words):
            split_word = paths[subset][word].split()
            if len(split_word) < 2 and paths[subset][word] in sentence:

                temporary.append(paths[subset][word])
        formatted_subsets.append(temporary)


    return formatted_subsets


def omit_subset(subset, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_len, classifier):

    x_left_omitted = x_left
    x_right_omitted = x_right
    x_left_omitted_len = x_left_len
    x_right_omitted_len = x_right_len

    dec_rule_left = np.zeros((1, 80))
    dec_rule_right = np.zeros((1, 80))
    dec_rule_left_len = x_left_len
    dec_rule_right_len = x_right_len

    if len(subset) > 0:
        # get word ID's for subset
        #word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_id_mapping = classifier.word_id_mapping
        amount_attributes = len(subset)
        word_ids = np.array([])


        for index in range(amount_attributes):
            if subset[index] in word_id_mapping:
                word_of_subset_id = classifier.word_id_mapping[subset[index]]
                word_ids = np.append(word_ids, word_of_subset_id)
        #print("These are the word ID's of my subset", word_ids)

        amount_of_ids = len(word_ids)

        for element in range(amount_of_ids):  # check if word ID in x_left, if yes set ID to 0

            if word_ids[element] in x_left_omitted:

                result = np.where(x_left_omitted == word_ids[element])
                x_left_omitted = np.delete(x_left_omitted, result[-1])
                #print("Found here", result[-1])
                count_found = len(result[-1])
                #print("Found in this many elements:", count_found)
                for i in range(count_found):
                    x_left_omitted = np.append(x_left_omitted, 0)
                x_left_omitted = np.array([x_left_omitted])
                x_left_omitted_len = x_left_omitted_len - count_found

                dec_rule_left[0, element] = word_ids[element]
                dec_rule_left_len = x_left_len - x_left_omitted_len

        #for element in range(amount_of_ids):  # check if word ID in x_right, if yes set ID to 0

            if word_ids[element] in x_right_omitted:
                result = np.where(x_right_omitted == word_ids[element])
                x_right_omitted = np.delete(x_right_omitted, result[-1])
                count_found = len(result[-1])
                for k in range(count_found):
                    x_right_omitted = np.append(x_right_omitted, 0)
                x_right_omitted = np.array([x_right_omitted])
                x_right_omitted_len = x_right_omitted_len - 1

                dec_rule_right[0, element] = word_ids[element]
                dec_rule_right_len = x_right_len - x_right_omitted_len
        #print("Length dec rule ", dec_rule_right_len)


    return x_left_omitted, x_left_omitted_len, x_right_omitted, x_right_omitted_len, y_true, target_word, target_len, dec_rule_right, dec_rule_right_len, dec_rule_left, dec_rule_left_len


def get_pred_difference(pred_b, prob_b, x_left_omitted, x_left_omitted_len, x_right_omitted, x_right_omitted_len,\
                        y_true, target_word, target_len, classifier):
    """

    :param pred_b: prediction made by black box model b
    :param prob_b: probabilities computed by b
    :param omitted_instance: instance (subset) we want to omit
    :param classifier: our black box model
    :return: prediction difference in probabilities
    """

    pred_omission, prob_omission = classifier.get_prob(x_left_omitted, x_left_omitted_len, x_right_omitted,
                                                       x_right_omitted_len, y_true, target_word, target_len)
    pred_diff = prob_b - prob_omission

    print("Probabilities for our 'neighbor': ", prob_omission)
    #print('Checking that probabilities sum up to 1: ', sum(prob_omission))
    #print("This is prediction difference: ", pred_diff)
    #print("This is prediction of 'neighbor': ", pred_omission)
    sent_diff = pred_b - pred_omission
    # print(sent_diff) #hier heb je niet zoveel aan
    return pred_diff, pred_omission


def get_tree_fidelity(rel_subset_pred, tree_pred):
    amount_pred = len(rel_subset_pred)
    count = 0
    for i in range(amount_pred):
        if rel_subset_pred[i] == tree_pred[i]:
            count = count + 1
    return count/amount_pred


def relevance_subsets(neighbor_probs, n_sentiments, temp_subsets, pred_x, labels, rule_pred):
    eff_pred_diff = np.array([])
    # rule_set = np.zeros((4, 3))
    #for sentiment in range(n_sentiments):
    diff = neighbor_probs[:, pred_x]
    max_pos_diff = np.max(diff)
    max_eff_diff = np.min(diff)
    #if abs(max_neg_diff) > max_pos_diff:
     #   max_eff_diff = max_neg_diff
    #else:
     #   max_eff_diff = max_pos_diff
    print("Max effect op sentiment: ", pred_x, max_eff_diff)
    eff_pred_diff = np.append(eff_pred_diff, max_eff_diff)
    max_effect_neg = np.where(diff == max_eff_diff)
    # print(max_effect_neg)
    # print("Index van max effect: ", max_effect_neg[-1])
    count = 0
    for ding in range(len(max_effect_neg[-1])):
        important_subset = temp_subsets[(max_effect_neg[-1][ding])]
        print("Subset met meeste effect op sentiment ", pred_x, important_subset)
        if labels[max_effect_neg[-1][ding]] == rule_pred[max_effect_neg[-1][ding]]:
            count = count + 1
    fidelity = count/len(max_effect_neg[-1])
    print('Fidelity for this rule: ', fidelity)
    return important_subset, fidelity


def get_instance_fid(rule_pred, labels):
    amount_pred = len(labels)
    count = 0
    for prediction in range(amount_pred):
        if labels[prediction] == rule_pred[prediction]:
            count = count + 1
    return count/amount_pred

if __name__ == '__main__':
    evaluation_main()
