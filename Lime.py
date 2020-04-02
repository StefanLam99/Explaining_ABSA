from classifier import *
from config import *
from loadData import getStatsFromFile
from utils import compare_preds
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.utils import check_random_state
import time
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)


def main():#initialisation of inputs:
    year = 2016
    model = 'Olaf'
    case = 'all' #correct, incorrect or nothing for all
    isWSP = False
    batch_size = 500 #we have to implement a batch size to get the predictions of the perturbed instances
    num_samples = 5000 #has to be divisible by batch size
    seed = 2020
    width = 0.5
    K = 5 # number of coefficients to check
    B = 10 # number of instances to get

    if year == 2015:
        input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
        model_path = 'trainedModelOlaf/2015/-12800'
    elif year==2016:
        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
        model_path = 'trainedModelOlaf/2016/-18800'

    f = classifier(input_file, model_path, year)
    dict_correct, dict_incorrect = f.get_split_instances()
    if case == 'correct':
        dict = dict_correct  # change depending on which case you want to know
    elif case == 'incorrect':
        dict = dict_incorrect
    else:
        dict = f.get_Allinstances()

    if(isWSP):
        write_path = 'data/Lime/WSP' + model + case + str(year)
    else:
        write_path = 'data/Lime/SP' + model + case + str(year)
    r = check_random_state(seed)


    #Start of LIME implementatino
    hit_left = np.array([])
    hit_right = np.array([])
    fid_left = []
    fid_right = []
    fid_all = []
    left_coefs = []
    right_coefs = []
    left_words = []
    right_words = []
    targets = []
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
    pred, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                               size)


    all_words =[]
    all_coefs = []
    start = time.time()
    with open(write_path + '.txt', 'w') as results:
        for index in range(size):
            #Getting the perturbations instances:

            x_inverse_left, x_lime_left, x_lime_left_len = lime_perturbation(r, x_left[index], x_left_len[index], num_samples)
            x_inverse_right, x_lime_right, x_lime_right_len = lime_perturbation(r, x_right[index], x_right_len[index], num_samples)

            target_lime_word = np.tile(target_word[index],(num_samples,1))
            target_lime_word_len = np.tile(target_words_len[index],(num_samples))
            y_lime_true = np.tile(y_true[index],(num_samples,1))

            #predicting the perturbations
            predictions,probabilities = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                       y_lime_true,target_lime_word, target_lime_word_len, batch_size, num_samples)


            #Getting the weights for each feature for left and right context
            weights_left = np.ones(num_samples)
            weights_right = np.ones(num_samples)
            if x_left_len[index] > 0:
                weights_left = get_weights(f, x_left[index],x_lime_left,x_left_len[index], x_lime_left_len,width)
            if x_right_len[index] > 0:
                weights_right = get_weights(f, x_right[index],x_lime_right,x_right_len[index],x_lime_right_len,width)

            #models
            model_left = LinearRegression(fit_intercept=False)
            model_right = LinearRegression(fit_intercept=False)

            #fitting the regression model with a constant
            constant = np.ones((num_samples, 1))
            if(x_left_len[index] > 0):
                xleft = np.concatenate((constant, x_inverse_left),axis=1)
                model_left.fit(xleft , predictions, sample_weight=weights_left)
                hit_left = np.append(hit_left,model_left.predict(np.concatenate((np.ones(1),x_inverse_left[0])).reshape(1,-1))[0])
                fidelity_left = model_left.predict(np.concatenate((constant, x_inverse_left),axis=1))
            else:
                xleft = constant
                model_left.fit(xleft, predictions)
                hit_left = np.append(hit_left,model_left.predict(np.ones(1).reshape(1,-1))[0])
                fidelity_left = model_left.predict(constant)

            if(x_right_len[index] > 0):
                xright = np.concatenate((constant, x_inverse_right),axis=1)
                model_right.fit(xright , predictions, sample_weight=weights_right)
                hit_right = np.append(hit_right,model_right.predict(np.concatenate((np.ones(1),x_inverse_right[0])).reshape(1,-1))[0])
                fidelity_right = model_right.predict(np.concatenate((constant, x_inverse_right),axis=1))
                xall = np.concatenate((xleft, xright), axis=1)
                coefsall = np.append(model_left.coef_, model_right.coef_)
                fidelity_all = np.matmul(xall, coefsall)
            else:
                xright = constant
                model_right.fit(xright, predictions)
                hit_right = np.append(hit_right, model_right.predict(np.ones(1).reshape(1,-1))[0])
                fidelity_right = model_right.predict(constant)
                xall = xleft
                coefsall = model_left.coef_
                print(coefsall)
                fidelity_all = np.matmul(xall, coefsall)




            #words:
            left_words.append(np.array(['C'] + f.get_String_Sentence(x_lime_left[0])))
            right_words.append(np.array(['C'] + f.get_String_Sentence(x_lime_right[0])))
            all_words.append(np.append(left_words[index], right_words[index]))
            targets.append(f.get_String_Sentence(target_word[index]))

            #coeffs
            left_coefs.append(model_left.coef_)
            right_coefs.append(model_right.coef_)
            all_coefs.append(np.append(model_left.coef_, model_right.coef_))


            #getting fidelity
            fidelity_all[fidelity_all > 0] = '1.0'
            fidelity_all[ fidelity_all <= 0] = '-1.0'
            _, fidel = compare_preds(predictions, fidelity_all)
            fid_all.append(fidel)

            fidelity_left[fidelity_left > 0] = '1.0'
            fidelity_left[ fidelity_left <= 0] = '-1.0'
            _, fidel = compare_preds(predictions, fidelity_left)
            fid_left.append(fidel)

            fidelity_right[fidelity_right > 0] = '1.0'
            fidelity_right[ fidelity_right <= 0] = '-1.0'
            _, fidel = compare_preds(predictions, fidelity_right)
            fid_right.append(fidel)





            results.write('Instance ' + str(index) +':' +'\n')
            results.write('True Label: ' + str(true_label[index]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('Left coefs: ' + str(left_coefs[index]) + '\n')
            results.write('Left words: ' + str(left_words[index]) + '\n')
            results.write('Right coefs: ' + str(right_coefs[index]) + '\n')
            results.write('Right words: ' + str(right_words[index]) + '\n')
            results.write('Target words: ' + str(targets[index]) + '\n')
            results.write('\n')
    results.close()

    ##getting the B instances according to (W)SP
    left_coefs_k = []
    right_coefs_k = []
    all_coefs_k = []

    e_ij_left = []
    e_ij_right = []
    e_ij = []

    left_words_k = []
    right_words_k = []
    all_words_k = []

    left_coefs = np.array(left_coefs)
    right_coefs = np.array(right_coefs)
    print(left_coefs)


    left_words = np.array(left_words)
    right_words = np.array(right_words)


    dict_I_left = {} #dict containing the global importance of all features
    dict_I_right = {} #dict containing the global importance of all features
    dict_I = {}
    for i in range(size):
        #coefficients are K or equal to the length of the sentence
        if(K > int(x_left_len[i])):
            lk = int(x_left_len[i])
        else:
            lk = K

        if(K > int(x_right_len[i])):
            rk = int(x_right_len[i])
        else:
            rk = K

        l_coefs = np.absolute(left_coefs[i] - left_coefs[i][0]) #beta_j - beta_0
        r_coefs = np.absolute(right_coefs[i] - right_coefs[i][0])
        left_maxargs = np.argpartition(l_coefs, -lk)[-lk:]
        right_maxargs = np.argpartition(r_coefs, -rk)[-rk:]

        #the k |e_ij|'s
        e_ij_left.append(l_coefs[left_maxargs])
        e_ij_right.append(r_coefs[right_maxargs])
        e_ij.append(np.append(l_coefs[left_maxargs], r_coefs[right_maxargs]))
        #the normal coefficients
        left_coefs_k.append(left_coefs[i][left_maxargs])
        right_coefs_k.append(right_coefs[i][right_maxargs])
        all_coefs_k.append(np.append(left_coefs[i][left_maxargs] - left_coefs[i][left_maxargs][0],
                                     right_coefs[i][right_maxargs]-right_coefs[i][right_maxargs][0]))

        left_words_k.append(left_words[i][left_maxargs])
        right_words_k.append(right_words[i][right_maxargs])
        all_words_k.append(np.append(left_words_k[i], right_words_k[i]))

        #getting the global importance for each feature
        for j, word in enumerate(left_words_k[i]):
            if(inDict(dict_I_left,word)):
               dict_I_left[word] += e_ij_left[i][j]
            else:
                dict_I_left[word] = e_ij_left[i][j]

        for j, word in enumerate(right_words_k[i]):
            if (inDict(dict_I_right, word)):
                dict_I_right[word] += e_ij_right[i][j]
            else:
                dict_I_right[word] = e_ij_right[i][j]

        for j, word in enumerate(all_words_k[i]):
            if (inDict(dict_I, word)):
                dict_I[word] += e_ij[i][j]
            else:
                dict_I[word] = e_ij[i][j]

    picked_instances_left = WSP(dict_I_left, left_words_k, left_coefs_k, B, isWSP)
    picked_instances_right = WSP(dict_I_right, right_words_k, right_coefs_k, B, isWSP)
    picked_instances_all = WSP(dict_I, all_words_k, all_coefs_k, B, isWSP)



    with open(write_path + 'all' + '.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(np.flip(right_words[i])) + '\n')
            results.write('\n')

            results.write('K coefs: ' + str(all_coefs_k[i]) + '\n')
            results.write('K influences: ' + str(np.array(all_coefs_k[i]) - all_coefs[i][0]) + '\n')
            results.write('K words: ' + str(left_words_k[i]) + str(right_words_k[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(all_coefs[i]) + '\n')
            results.write('words: ' + str(all_words[i]) + '\n')

            results.write('\n')
            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
            results.write('\n')
        hit_all = hit_left + hit_right
        hit_all[hit_all > 0] = 1
        hit_all[hit_all <= 0] = -1
        correctAll, hitall = compare_preds(pred, hit_all)
        results.write(str(hitall) + '\n')
        results.write('Hit Rate All measure:' + '\n')
        results.write('Correct: ' + str(correctAll) + ' hit rate: ' + str(hitall) + '\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fid_all)
        std = np.std(fid_all)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))
    #writing results for (W)SP
    with open(write_path + 'left' + '.txt', 'w') as results:
        for i in picked_instances_left:
            results.write('picked left instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(np.flip(right_words[i])) + '\n')
            results.write('\n')

            results.write('K coefs left: ' + str(left_coefs_k[i]) + '\n')
            results.write('K influences left: ' + str(np.array(left_coefs_k[i]) - left_coefs[i][0]) + '\n')
            results.write('K words left: ' + str(left_words_k[i]) + '\n')
            results.write('\n')
            results.write('Left coefs: ' + str(left_coefs[i]) + '\n')
            results.write('Left words: ' + str(left_words[i]) + '\n')
            results.write('\n')
            results.write('K coefs right: ' + str(right_coefs_k[i]) + '\n')
            results.write('K influences right: ' + str(np.array(right_coefs_k[i]) - right_coefs[i][0]) + '\n')
            results.write('K words right: ' + str(right_words_k[i]) + '\n')
            results.write('\n')
            results.write('Right coefs: ' + str(right_coefs[i]) + '\n')
            results.write('Right words: ' + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
            results.write('\n')

        hit_left[hit_left>0] = 1
        hit_left[hit_left<=0] = -1
        correctleft, hitleft = compare_preds(pred, hit_left)
        results.write(str(hit_left) + '\n')
        results.write('Hit Rate Left measure:' + '\n')
        results.write('Correct: ' + str(correctleft) + ' Hit rate: ' + str(hitleft) + '\n')
        results.write('Fidelity Left measure: ' + '\n')
        mean = np.mean(fid_left)
        std = np.std(fid_left)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))
    results.close()
    with open(write_path + 'right' + '.txt', 'w') as results:
        for i in picked_instances_right:
            results.write('picked right instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(np.flip(right_words[i])) + '\n')
            results.write('\n')

            results.write('K coefs left: ' + str(left_coefs_k[i]) + '\n')
            results.write('K influences left: ' + str(np.array(left_coefs_k[i]) - left_coefs[i][0]) + '\n')
            results.write('K words left: ' + str(left_words_k[i]) + '\n')
            results.write('\n')
            results.write('Left coefs: ' + str(left_coefs[i]) + '\n')
            results.write('Left words: ' + str(left_words[i]) + '\n')
            results.write('\n')
            results.write('K coefs right: ' + str(right_coefs_k[i]) + '\n')
            results.write('K influences right: ' + str(np.array(right_coefs_k[i]) - right_coefs[i][0]) + '\n')
            results.write('K words right: ' + str(right_words_k[i]) + '\n')
            results.write('\n')
            results.write('Right coefs: ' + str(right_coefs[i]) + '\n')
            results.write('Right words: ' + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
            results.write('\n')

        hit_right[hit_right>0] = 1
        hit_right[hit_right<=0] = -1
        correctright, hitright = compare_preds(pred, hit_right)
        results.write(str(hit_right) + '\n')
        results.write('Hit Rate Right measure:' + '\n')
        results.write('Correct: ' + str(correctright) + ' Hit rate: ' + str(hitright) + ' \n')
        results.write('Fidelity Right measure: ' + '\n')
        mean = np.mean(fid_right)
        std = np.std(fid_right)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    results.close()



    end = time.time()
    print("It took: " + str(end - start) + 'seconds')



    print(picked_instances_left)
    print(picked_instances_right)






def WSP(dict_I, words, coefs, B, isWSP):
    """

    :param dict_I:
    :param words: all the instances
    :param coefs: the absolute weights |e_ij|
    :param B: max number of instances to pick
    :return:
    """
    picked_instances = []

    while(len(picked_instances) < B):
        c_max = -1
        picked_instance = -1
        for i, sentence in enumerate(words):
            c = 0
            if(isWSP):
                for j, word in enumerate(sentence):
                    c += coefs[i][j] * np.sqrt(dict_I[word]) #coverage with weights
            else:
                for j, word in enumerate(sentence):
                    c += np.sqrt(dict_I[word])  # coverage without weights SP
            if(c > c_max): ## this is the max coverage according to a greedy algorithm
                picked_instance = i
                c_max = c

            for s in words[picked_instance]:
                dict_I[s] = 0 ## we already incorporated these words in the picked instances

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
        return np.zeros((num_samples,FLAGS.max_sentence_len)).astype(int)\
            , np.zeros((num_samples,FLAGS.max_sentence_len)).astype(int)\
            , np.zeros(num_samples).astype(int)
    else:
        sample = random.randint(0, x_len, num_samples-1)

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



if __name__ == '__main__':
    main()


