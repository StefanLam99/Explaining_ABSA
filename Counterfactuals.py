from decisionTree import *
import time
from utils import *


def diff_sc(instance, q):
    """
    method to get the number of split condtions satisfying q
    but not the instance
    :param instance:
    :param q:
    :return:
    """

    same_sc = 0
    qlen = len(q)
    for word in instance:
        for q_word in q:
            if(word == q_word):
                same_sc += 1

    return qlen - same_sc

def get_counterfactuals(instance, root_leaf_paths, true_label):
    """
    Method to get the counterfactuals paths from all root leaf paths
    given the true lavel of the instance
    :param instance:
    :param root_leaf_paths: dictionary with all root leaf paths
    :param true_label: the true label of the instance
    :return: a dictionary containgin the correspondsing counterfactuals
    """
    diff_paths = {}
    counterfactuals = {}
    diff_keys = []
    for key in root_leaf_paths.keys():
        if key != str(true_label):
            diff_paths[key] = root_leaf_paths[key]
            diff_keys.append(key)
            counterfactuals[key] = []

    min = float('inf')

    for key in diff_keys:#a bit ugly, but gets the job done
        for q in diff_paths[key]:
            qlen = diff_sc(instance, q)
            if qlen < min:
                for key2 in diff_keys:
                    counterfactuals[key2] = []
                min = qlen
                counterfactuals[key] = [q]
            elif qlen == min:
                counterfactuals[key] += [q]
    return counterfactuals

def get_cfInstance(instance, counterfactuals):
    """
    Method to get the counterfactuals instance of a counterfactual
    :param instance: the instance with no label
    :param counterfactuals: see get_counterfactuals
    :return: a dictionary with the cf instances
    """
    cf_instances = {}
    for key in counterfactuals.keys():
        cf_instances[key] = []
        for cf in counterfactuals[key]:
            temp = []
            for word in instance:
                inCF = True
                for cf_word in cf:
                    split_words = cf_word.split()
                    if len(split_words) >= 2:
                        if(split_words[1] == word):
                            inCF = False
                if(inCF):

                    temp.append(word)
                else:
                    temp.append('not ' + word)

            cf_instances[key] += [temp]
    return cf_instances
def make_full_sentence(left_sentence, right_sentence):
    full_sentence = []
    for i in range(len(left_sentence)):
        temp = left_sentence[i].copy()
        temp.pop()
        full_sentence.append(temp + right_sentence[i])

    return full_sentence

def get_fid_instance(rules, sentences):
    """
    gets the local fidelity with a set of rules as a dictionary with keys '0','1','-1'
    :param rules:
    :param sentences: the perturbed instances
    :return:
    """
    size = 0
    correct = 0

    def match(sentence, path):
        flag = False
        for path_word in path:
            flag = False
            for word in sentence:
                s = path_word.split()
                if (len(s) >= 2):
                    flag = True
                    if (s[1] == word):
                        flag = False
                        break
                else:
                    if (word == path_word):
                        flag = True

            if (not flag): break
        return flag


    for temp in sentences:
        flag = False
        sentence = temp.copy()
        pred = sentence.pop()
        for key in rules.keys(): #check for all paths the first one that is satisfied should be one
            for path in rules[key]:
                if(match(sentence, path)):
                    size +=1
                    flag = True

                    if (int(key) == int(float(pred))):
                        correct += 1
                if(flag):
                    break
            if(flag):
                break
    return correct, size

def inDict(dict, key):
    for keys in dict.keys():
        if key == keys:
            return True
    return False

def get_cf_instance_stats(f, cf_instance, index):
    """
    Gets the number of cf instances, and the number of correct predicted cf instances between f and c.
    :param f: classifier
    :param cf_instance: dict
    :param index: index of the instance
    :return:
    """
    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_word_len = f.get_instance(index)
    r_len = int(x_right_len)
    l_len= int(x_left_len)
    nInstances = 0
    correct = 0
    for key in cf_instance.keys():
        if (len(cf_instance[key]) <= 0):
            break

        for instance in cf_instance[key]:
            nInstances += 1
            temp_instance = np.zeros(len(instance))

            for i, word in enumerate(instance):
                if(len(word.split()) <2 and inDict(f.word_id_mapping, word)):
                    temp_instance[i] = f.word_id_mapping[word]
                else:
                    temp_instance[i] = 0

            instance_left = np.zeros(FLAGS.max_sentence_len)
            instance_left[0:l_len] = temp_instance[0:l_len]
            instance_left = instance_left.reshape((1, FLAGS.max_sentence_len))

            instance_right = np.zeros(FLAGS.max_sentence_len)
            instance_right[0:r_len] = temp_instance[l_len:l_len+r_len]
            instance_right = instance_right.reshape((1, FLAGS.max_sentence_len))

            pred, _ = f.get_prob(instance_left, x_left_len, instance_right, x_right_len,y_true,target_word, target_word_len)
            if(int(key) == pred):
                correct +=1

    return correct, nInstances




def main():
    year = 2016
    model = 'Maria' # or 'Olaf'
    model = 'Olaf'
    num_samples = 5000
    batch_size = 200
    r = check_random_state(2020)
    if model == 'Olaf':
        write_path = 'data/Counterfactuals' + model + str(2016)
    elif model == "Maria":
        write_path = 'data/Counterfactuals' + model + str(2016)

    begin = time.time()
    f = classifier(model=model)

    correct_full = 0
    correct_cf_instances = 0
    size_cf_instances = 0
    fidelity = []
    fid_cf = []
    fid_tree = []
    size = f.size
    for index in range(size):

        ## getting data and building trees

        classifier_pred, true_label, pred_c, x_inverse_left, left_sentences, x_inverse_right, \
        right_sentences = data(f, r, num_samples,batch_size,index=index)



        #full
        full_sentences = make_full_sentence(left_sentences, right_sentences)
        features_full = full_sentences[0]
        root_full = build_tree(full_sentences, features_full, 0)
        tree_full = Tree(root_full)

        pred_full = classify(full_sentences[0], root_full)
        if (pred_full == str(int(classifier_pred))):
            correct_full += 1

        counter = 0
        for i, sentence in enumerate(full_sentences):
            pred = classify(sentence, root_full)
            if(int(pred) == int(float(pred_c[i]))):
                counter +=1

        fidelity.append(counter/num_samples)

        instance = full_sentences[0].copy()
        instance.pop()  # get rid of the label
        root_leaf_paths = tree_full.get_paths()
        counterfactuals = get_counterfactuals(instance, root_leaf_paths, true_label)
        correct_tree, size_tree = get_fid_instance(root_leaf_paths, full_sentences)

        if(size_tree >0):
            fid_tree.append(correct_tree/size_tree)

        correct_cf, size_cf = get_fid_instance(counterfactuals, full_sentences)
        if(size_cf >0):
            fid_cf.append(correct_cf/size_cf)

        cf_instance = get_cfInstance(instance, counterfactuals)
        correct, nInstances = get_cf_instance_stats(f, cf_instance, index)
        correct_cf_instances += correct
        size_cf_instances += nInstances





    #cf_instance = get_cfInstance(instance, counterfactuals)


    end = time.time()
    seconds = end - begin
    '''
    print('tree: ' + str(fid_tree))
    print('counterfactual: ' + str(fid_cf))
    print(size_tree)
    print(size_cf)
    print_tree(root_full)
    print(root_leaf_paths)
    print(counterfactuals)

    print(true_label)
    '''



    with open(write_path + '.txt', 'w') as results:
        results.write('Hit Rate Instances: ' + str(correct_full/size) + '\n')
        results.write('Hit Rate Counterfactual Instances: ' + str(correct_cf_instances/size_cf_instances) + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('\n')
        results.write('Fidelity Measure Decision Tree: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')
        mean = np.mean(fid_tree)
        std = np.std(fid_tree)
        results.write('Fidelity Measure Decision Tree Rules: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')
        mean = np.mean(fid_cf)
        std = np.std(fid_cf)
        results.write('Fidelity Measure Counterfactuals Rules: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')



    print('It took: ' + str(seconds) + ' seconds')
if __name__ == '__main__':
    main()


