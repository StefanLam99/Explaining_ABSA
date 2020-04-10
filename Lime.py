from classifier import *
from config import *
from loadData import getStatsFromFile
from utils import compare_preds
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_random_state
import time
import numpy as np
from Anchor import get_perturbations
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['SPACY_WARNING_IGNORE'] = 'W008'
#np.set_printoptions(threshold=sys.maxsize)
def main2():
    begin = time.time()
    model = 'Maria'
    isWSP = False
    batch_size = 200 #we have to implement a batch size to get the predictions of the perturbed instances
    num_samples = 5000 #has to be divisible by batch size
    seed = 2020
    width = 1.0
    K = 5 # number of coefficients to check
    B = 10 # number of instances to get
    input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
    model_path = 'trainedModelOlaf/2016/-18800'
    f = classifier(model)
    dict = f.get_Allinstances()
    r = check_random_state(seed)
    if(isWSP):
        write_path = 'data/Lime/WSPfh' + model + str(2016) + 'final'
    else:
        write_path = 'data/Lime/SPfh' + model + str(2016) + 'final'

    #Estimating Lime with multinominal logistic regression
    fidelity = []
    correct_hit = 0
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']
    true_label = dict['true_label']
    pred = dict['pred']
    size = dict['size']
    left_words = []
    right_words = []
    all_words = []

    targets = []
    x_len = []
    coefs = []
    pred_b, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size, size)
    with open(write_path + '.txt', 'w') as results:
        for index in range(size):
            x_inverse_left, x_lime_left, x_lime_left_len = lime_perturbation(r, x_left[index], x_left_len[index], num_samples)
            x_inverse_right, x_lime_right, x_lime_right_len = lime_perturbation(r, x_right[index], x_right_len[index],
                                                                                num_samples)

            target_lime_word = np.tile(target_word[index], (num_samples, 1))
            target_lime_word_len = np.tile(target_words_len[index], (num_samples))
            y_lime_true = np.tile(y_true[index], (num_samples, 1))



            # predicting the perturbations
            pred_c, probabilities = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                                       y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                                       num_samples)

            neg_labels = labels(pred_c)
            # Getting the weights
            x_w = np.append(x_left[index][0:x_left_len[index]], x_right[index][0:x_right_len[index]])
            x_w_len = x_left_len[index] + x_right_len[index]
            x_len.append(x_w_len)
            x_lime_len = x_lime_left_len + x_lime_right_len
            x_lime = np.concatenate((x_lime_left, x_lime_right), axis=1)
            weights_all = get_weights(f, x_w, x_lime, x_w_len, x_lime_len, width)


            model_all = LogisticRegression(multi_class='ovr', solver = 'newton-cg')

            n_neg_labels = len(neg_labels)

            x_all = np.concatenate((x_inverse_left,x_inverse_right), axis=1)

            if n_neg_labels > 0:
                for label in neg_labels:
                    pred_c = np.append(pred_c,label)
                    x_all = np.concatenate((x_all, np.zeros((1,x_left_len[index] + x_right_len[index]))), axis = 0)
                    weights_all = np.append(weights_all,0)

                model_all.fit(x_all, pred_c, sample_weight=weights_all)
                pred_c = pred_c[:-n_neg_labels]
                x_all = x_all[:-n_neg_labels,:]
            else:
                model_all.fit(x_all, pred_c, sample_weight=weights_all)

            yhat = model_all.predict(x_all)
            if(int(yhat[0]) == int(pred_b[index])):
                correct_hit+=1
            _, acc = compare_preds(yhat, pred_c)
            fidelity.append(acc)

            # words:
            left_words.append(f.get_String_Sentence(x_lime_left[0]))
            right_words.append(f.get_String_Sentence(x_lime_right[0]))
            all_words.append(f.get_String_Sentence(x_lime_left[0]) + f.get_String_Sentence(x_lime_right[0]))
            targets.append(f.get_String_Sentence(target_word[index]))


            coefs.append(model_all.coef_)
            intercept = model_all.intercept_
            classes = model_all.classes_

            results.write('Instance ' + str(index) + ':' + '\n')
            results.write(
                'True Label: ' + str(true_label[index]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('\n')
            results.write('Intercept: ' + str(intercept) + '\n')
            results.write('\n')
            results.write('Left words: ' + str(left_words[index]) + '\n')
            results.write('\n')
            temp = right_words.copy()
            temp[index].reverse()
            results.write('Right words: ' + str(temp[index]) + '\n')
            results.write('\n')
            results.write('All words: ' + str(all_words[index]) + '\n')
            results.write('Target words: ' + str(targets[index]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')

        neg_coefs_k = []
        neu_coefs_k = []
        pos_coefs_k = []
        all_coefs_k = []
        e_ij = []
        sum_coefs_k = []

        all_words_k = []
        dict_I = {}
        for i in range(size):
            K = 4
            if(K > int(x_len[i])):
                K = int(x_len[i])

            ##getting the B instances according to (W)SP
            neg_coefs = coefs[i][0]
            neu_coefs = coefs[i][1]
            pos_coefs = coefs[i][2]


            sum_coefs = np.zeros(len(neg_coefs))
            for k in range(len(neg_coefs)):
                sum_coefs[k] += np.absolute(neg_coefs[k]) + np.absolute(pos_coefs[k]) + np.absolute(neg_coefs[k])

            coefs_maxargs = np.argpartition(sum_coefs, -K)[-K:]
            neg_coefs_k.append(neg_coefs[coefs_maxargs])
            neu_coefs_k.append(neu_coefs[coefs_maxargs])
            pos_coefs_k.append(pos_coefs[coefs_maxargs])


            sum_coefs_k.append(sum_coefs[coefs_maxargs])
            e_ij.append(sum_coefs[coefs_maxargs])

            all_coefs_k.append([neg_coefs_k[i], neu_coefs_k[i], pos_coefs_k[i]])

            temp = np.array(all_words[i])
            all_words_k.append(temp[coefs_maxargs])

            for j, word in enumerate(all_words_k[i]):
                if(inDict(dict_I, word)):
                    dict_I[word] += e_ij[i][j]
                else:
                    dict_I[word] = e_ij[i][j]

            results.write('Instance: ' + str(i))
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')
        results.close()

    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, isWSP)



    with open(write_path + 'K_instances' + '.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')



            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    end = time.time()
    print('It took: ' + str(end-begin) + ' Seconds')

def main3():
    begin = time.time()
    model = 'Olaf'
    #isWSP = False
    batch_size = 200 #we have to implement a batch size to get the predictions of the perturbed instances
    num_samples = 5000 #has to be divisible by batch size
    seed = 2020
    width = 1.0
    K = 5 # number of coefficients to check
    B = 10 # number of instances to get

    f = classifier(model)
    dict = f.get_Allinstances()
    r = check_random_state(seed)
    write_path ='data/Lime2/' + model + str(2016) + 'final'
    '''
    if(isWSP):
        write_path = 'data/Lime2/WSP' + model + str(2016) + 'final'
    else:
        write_path = 'data/Lime2/SP' + model + str(2016) + 'final'
    '''
    #Estimating Lime with multinominal logistic regression
    n_all_features = len(f.word_id_mapping)
    fidelity = []
    correct_hit = 0
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']
    true_label = dict['true_label']
    pred = dict['pred']
    size = dict['size']
    left_words = []
    right_words = []
    all_words = []

    targets = []
    x_len = []
    coefs = []
    size = 30

    pred_b, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size, size)
    original_x = []
    with open(write_path + '.txt', 'w') as results:
        for index in range(size):
            pertleft, instance_sentiment, text, _, x = get_perturbations(True, False, f, index, num_samples)
            pertright, instance_sentiment, text, _, x = get_perturbations(False, True, f, index, num_samples)
            orig_left_x = x_left[index]
            orig_right_x = x_right[index]
            Z = np.zeros((num_samples, n_all_features))
            X = np.zeros((n_all_features))
            print(orig_right_x)
            print(orig_left_x)
            X[orig_left_x] += 1
            X[orig_right_x] += 1
            X = X.reshape(1,-1)
            predictions_f = []
            x_lime = np.zeros((num_samples,x_left_len[index] + x_right_len[index]))
            for i in range(num_samples):

                x_left_ids = f.to_input(pertleft[i].split())
                x_right_ids = f.to_input(pertright[i].split())
                x_lime[i,0:x_left_len[index]+x_right_len[index]] = np.append(x_left_ids[0][0:x_left_len[index]], x_right_ids[0][0:x_right_len[index]])
                Z[i, x_left_ids] += 1
                Z[i, x_right_ids] += 1
                pred_f, _ = f.get_prob(x_left_ids, x[1], x_right_ids, x[3], x[4], x[5], x[6])
                predictions_f.append(pred_f)



            neg_labels = labels(predictions_f)
            # Getting the weights
            orig_x = np.append(orig_left_x[0:x_left_len[index]], orig_right_x[0:x_right_len[index]])
            original_x.append(orig_x)
            orig_x_len = int(x_left_len[index] + x_right_len[index])
            x_len.append(orig_x_len)
            z_len = np.tile(orig_x_len, num_samples)
            x_lime = np.asarray(x_lime, int)
            print(x_lime)
            weights_all = get_weights(f, orig_x, x_lime, orig_x_len, z_len, width)


            model_all = LogisticRegression(multi_class='ovr', solver = 'newton-cg')

            n_neg_labels = len(neg_labels)



            if n_neg_labels > 0:
                for label in neg_labels:
                    predictions_f = np.append(predictions_f,label)
                    Z = np.concatenate((Z, np.zeros((1,n_all_features))), axis = 0)
                    weights_all = np.append(weights_all,0)

                model_all.fit(Z, predictions_f, sample_weight=weights_all)
                predictions_f = predictions_f[:-n_neg_labels]
                Z = Z[:-n_neg_labels,:]
            else:
                model_all.fit(Z, predictions_f, sample_weight=weights_all)

            yhat = model_all.predict(X)
            print('aaaaaaaaaaaaaaaah')
            print(yhat)
            print(int(yhat))
            print(pred_b[index])
            print(int(pred_b[index]))
            if(int(yhat[0]) == int(pred_b[index])):
                correct_hit+=1

            yhat = model_all.predict(Z)

            _, acc = compare_preds(yhat, predictions_f)
            fidelity.append(acc)

            # words:
            left_words.append(f.get_String_Sentence(orig_left_x))
            right_words.append(f.get_String_Sentence(orig_right_x))
            all_words.append(f.get_String_Sentence(orig_left_x) + f.get_String_Sentence(orig_right_x))
            targets.append(f.get_String_Sentence(target_word[index]))


            coefs.append(model_all.coef_)
            intercept = model_all.intercept_
            classes = model_all.classes_

            results.write('Instance ' + str(index) + ':' + '\n')
            results.write(
                'True Label: ' + str(true_label[index]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('\n')
            results.write('Intercept: ' + str(intercept) + '\n')
            results.write('\n')
            results.write('Left words: ' + str(left_words[index]) + '\n')
            results.write('\n')
            temp = right_words.copy()
            temp[index].reverse()
            results.write('Right words: ' + str(temp[index]) + '\n')
            results.write('\n')
            results.write('All words: ' + str(all_words[index]) + '\n')
            results.write('Target words: ' + str(targets[index]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')


        neg_coefs_k = []
        neu_coefs_k = []
        pos_coefs_k = []
        all_coefs_k = []
        e_ij = []
        sum_coefs_k = []
        all_words_k = []
        dict_I = {}

        for i in range(size):
            K = 4
            if(K > int(x_len[i])):
                K = int(x_len[i])

            ##getting the B instances according to (W)SP
            neg_coefs = coefs[i][0]
            neu_coefs = coefs[i][1]
            pos_coefs = coefs[i][2]


            sum_coefs = np.zeros(len(neg_coefs))
            for j in original_x[i]:
                sum_coefs[j] += np.absolute(neg_coefs[j]) + np.absolute(pos_coefs[j]) + np.absolute(neg_coefs[j])

            coefs_maxargs = np.argpartition(sum_coefs, -K)[-K:]
            neg_coefs_k.append(neg_coefs[coefs_maxargs])
            neu_coefs_k.append(neu_coefs[coefs_maxargs])
            pos_coefs_k.append(pos_coefs[coefs_maxargs])


            sum_coefs_k.append(sum_coefs[coefs_maxargs])

            e_ij.append(sum_coefs[coefs_maxargs])

            all_coefs_k.append([neg_coefs_k[i], neu_coefs_k[i], pos_coefs_k[i]])
            all_words_k.append(f.get_String_Sentence(coefs_maxargs))
            #temp = np.array(all_words[i])
            #all_words_k.append(temp[coefs_maxargs])

            for j, word in enumerate(all_words_k[i]):
                if(inDict(dict_I, word)):
                    dict_I[word] += e_ij[i][j]
                else:
                    dict_I[word] = e_ij[i][j]

            results.write('Instance: ' + str(i) + '\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')
        results.close()

    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, True)



    with open(write_path + 'B_instances' + 'WSP.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')



            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))


    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, False)

    with open(write_path + 'B_instances' + 'SP.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')



            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    end = time.time()
    print('It took: ' + str(end-begin) + ' Seconds')



def WSP(dict_I, words, coefs, B, isWSP):
    """

    :param dict_I:
    :param words: all the instances
    :param coefs: the absolute weights |e_ij|
    :param B: max number of instances to pick
    :return:
    """
    picked_instances = []
    dict_I_copy = dict_I.copy()

    while(len(picked_instances) < B):
        c_max = -1
        picked_instance = -1
        for i, sentence in enumerate(words):
            c = 0
            if(isWSP):
                for j, word in enumerate(sentence):
                    c += coefs[i][j] * np.sqrt(dict_I_copy[word]) #coverage with weights
            else:
                for j, word in enumerate(sentence):
                    c += np.sqrt(dict_I_copy[word])  # coverage without weights SP
            if(c > c_max): ## this is the max coverage according to a greedy algorithm
                picked_instance = i
                c_max = c

            for s in words[picked_instance]:
                dict_I_copy[s] = 0 ## we already incorporated these words in the picked instances

        picked_instances.append(picked_instance)

    return np.array(picked_instances)


def inDict(dict, key):
    for keys in dict.keys():
        if key == keys:
            return True
    return False


def lime_perturbation(random, x, x_len, num_samples):
    """
    Generates the neighborhood of a sentence unirformly, input are all sentences.
    :param random: randomState object
    :param x: array with sentences corresponding to id's
    :param x_len: length of x
    :param num_samples:
    :return: x_inverse: the interpretable perturbed instance
             x_lime: the perturbed instance, to feed dict
             x_lime_len: length of x_lime
    """

    if(x_len>1):
        sample = random.randint(1, x_len, num_samples-1)
    elif(x_len==0):#if there is no context
        return np.zeros((num_samples,x_len)).astype(int)\
            , np.zeros((num_samples,FLAGS.max_sentence_len)).astype(int)\
            , np.zeros(num_samples).astype(int)
    else:
        sample = random.randint(0, x_len+1, num_samples-1)

    features_range = range(x_len)

    #length
    x_lime_len = np.zeros(num_samples)
    x_lime_len[0] = x_len
    # perturbed interpretable instaces data
    x_inverse = np.ones((num_samples, x_len))
    x_inverse[0,:] = np.ones(x_len)

    x_lime = np.zeros((num_samples,FLAGS.max_sentence_len))

    x_lime[0, 0:x_len] = np.multiply(x[0:x_len], x_inverse[0, 0:x_len])

    for i, size in enumerate(sample, start=1):

        inactive = random.choice(features_range, size, replace=False)
        x_inverse[i, inactive] = 0
        x_lime[i, 0:x_len] = np.multiply(x[0:x_len], x_inverse[i,0:x_len])
        x_lime_len[i] = np.sum(x_inverse[i,0:x_len])
        x_inverse[i,:] = x_inverse[i,0:x_len]

    x_inverse = x_inverse.astype(int)
    x_lime = x_lime.astype(int)
    x_lime_len = x_lime_len.astype(int)
    return x_inverse,  x_lime, x_lime_len



def distance(x, z):
    """
    Cosine distance of two vectors x and z
    """
    return np.inner(x, z) / (np.linalg.norm(x) * np.linalg.norm(z))

def kernel(x,z, width):
    """
    Exponential kernel function with given width
    """
    return np.exp(-np.power(distance(x,z),2)/np.power(width,2))

def get_weights(f, x,z, x_len, z_len, width):
    """
    Gets the weights of the perturbed samples, input is the whole perturbed sample
    :param f: classifier
    :param x: sentence with id's
    :param z: sentence with id's
    :param x_len: length of x
    :param z_len: length of z
    :param width: width which determines the the size of the neighborhood
    :return: the weights for each perturbed instance (size = num_samples)
    """
    num_samples,_ = z.shape
    z = z[:,0:x_len]

    x_glove = f.get_GloVe_embedding(x, x_len)
    x_average = x_glove.sum(axis=1)/x_len

    weights = np.zeros(num_samples)

    for i in range(num_samples):
        z_glove = f.get_GloVe_embedding(z[i,:],z_len[i])
        z_average = (z_glove.sum(axis=1)/z_len[i])
        weights[i] = kernel(x_average,z_average,width)

    return weights

def countZeros(list):
    counter = 0
    for e in list:
        if int(e) ==0:
            counter+=1
    return counter

def labels(pred):
    flag = False
    labels = [-1,0,1]
    for e in pred:
        for label in labels:
            if int(e) == int(label):
                labels.remove(e)
    return labels
if __name__ == '__main__':
    #main()
    #main2()
    main3()
''' 
model = 'Olaf'
isWSP = False
batch_size = 200 #we have to implement a batch size to get the predictions of the perturbed instances
num_samples = 2000 #has to be divisible by batch size
seed = 2020
width = 1.0
K = 5 # number of coefficients to check
B = 10 # number of instances to get
input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
model_path = 'trainedModelOlaf/2016/-18800'
f = classifier(model)
dict = f.get_Allinstances()
fidelity = []
correct_hit = 0
x_left = dict['x_left']
x_left_len = dict['x_left_len']
x_right = dict['x_right']
x_right_len = dict['x_right_len']
target_word = dict['target']
target_words_len = dict['target_len']
y_true = dict['y_true']
true_label = dict['true_label']
pred = dict['pred']
size = dict['size']
left_words = []
right_words = []
all_words = []
neg_coefs_k = []
neu_coefs_k = []
pos_coefs_k = []
targets = []
x_len = []
coefs = []
index=4
r = check_random_state(seed)
index = 4

pred_b, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size, size)

x_inverse_left, x_lime_left, x_lime_left_len = lime_perturbation(r, x_left[index], x_left_len[index], num_samples)
x_inverse_right, x_lime_right, x_lime_right_len = lime_perturbation(r, x_right[index], x_right_len[index],
                                                                            num_samples)

target_lime_word = np.tile(target_word[index], (num_samples, 1))
target_lime_word_len = np.tile(target_words_len[index], (num_samples))
y_lime_true = np.tile([1,0,0], (num_samples, 1))


pred_c, probabilities = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                  y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                  num_samples)

for i in range(num_samples):
    pred, prob = f.get_prob(x_lime_left[i].reshape(1,-1), np.array([x_lime_left_len[i]]), x_lime_right[i].reshape(1,-1), np.array([x_lime_right_len[i]]),
                                  y_lime_true[i].reshape(1,-1), target_lime_word[i].reshape(1,-1), np.array([target_lime_word_len[i]]))
    
    print(y_lime_true[i])
    print(pred)
    print(x_lime_left[i])
    print(x_lime_right[i])
    print(target_lime_word[i])
    

print(y_true[index])
print(pred_c)
print(sum(pred_c))
print(x_inverse_left)
print(x_lime_left)
print(x_inverse_right)
print(x_lime_right)
'''
