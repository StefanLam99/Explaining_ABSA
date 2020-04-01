from decisionTree import *
import time



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
        temp = left_sentence[i]
        temp.pop()
        full_sentence.append(temp + right_sentence[i])

    return full_sentence

def main():
    year = 2016
    model = 'test'
    case = 'all'  # correct, incorrect or nothing for all
    index = 4
    num_samples = 5000
    batch_size = 500
    r = check_random_state(2020)
    if year == 2015:
        input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
        model_path = 'trainedModelOlaf/2015/-12800'
    elif year == 2016:
        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
        model_path = 'trainedModelOlaf/2016/-18800'

    begin = time.time()
    f = classifier(input_file=input_file, model_path=model_path, year=year)

    correct_full = 0
    correct_left = 0
    correct_right = 0

    for index in range(295):


        ## getting data and building trees

        classifier_pred, true_label, predictions, x_inverse_left, left_sentences, x_inverse_right, \
        right_sentences = data(f, r, year,model,case,num_samples,batch_size,index=index)

        #right
        features_right = right_sentences[0]
        root_right = build_tree(right_sentences, features_right, 0)
        tree_right = Tree(root_right)

        #left
        features_left = left_sentences[0]
        root_left = build_tree(left_sentences, features_left, 0)
        tree_right = Tree(root_left)

        #full
        full_sentences = make_full_sentence(left_sentences, right_sentences)
        features_full = full_sentences[0]
        root_full = build_tree(full_sentences, features_full, 0)
        tree_full = Tree(root_full)

        pred_right = classify(right_sentences[0], root_right)
        if(pred_right == str(int(classifier_pred))):
            correct_right +=1

        pred_left = classify(left_sentences[0], root_left)
        if (pred_left == str(int(classifier_pred))):
            correct_left += 1

        pred_full = classify(full_sentences[0], root_full)
        if (pred_full == str(int(classifier_pred))):
            correct_full += 1

        print(correct_right)
        print(correct_left)
        print(correct_full)





        ''' 
        instance = right_sentences[0]
        instance.pop()#get rid of the label

        counterfactuals = get_counterfactuals(instance, root_leaf_paths, true_label)

        cf_instance = get_cfInstance(instance, counterfactuals)
        print(counterfactuals)
        print(cf_instance)
        print(instance)
        '''
    end = time.time()
    seconds = end - begin
    print('Fidelity Full: ' + str(correct_full/295))
    print('Fidelity Right: ' + str(correct_right / 295))
    print('Fidelity Left: ' + str(correct_left / 295))
    print('It took: ' + str(seconds) + ' seconds')
if __name__ == '__main__':
    main()


