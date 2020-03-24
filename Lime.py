from classifier import *
from config import *
from loadData import getStatsFromFile
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.utils import check_random_state
import time
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)


def main():#initialisation of inputs:
    year = 2015

    if year == 2015:
        input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
        model_path = 'trainedModelOlaf/2015/-12800'
    elif year==2016:

        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
        model_path = 'trainedModelOlaf/2016/-18800'

    f = classifier(input_file, model_path, year)
    dict_correct, dict_incorrect = f.get_split_instances()
    batch_size = 200 #we have to implement a batch size to get the predictions of the perturbed instances
    num_samples = 5000 #has to be divisible by batch size
    seed = 2020
    dict = dict_correct #change depending on which case you want to know
    width = 0.5
    write_path = 'data/Lime/test2.txt'
    r = check_random_state(seed)


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
    start = time.time()
    with open(write_path, 'w') as results:
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
            model_left = LinearRegression()
            model_right = LinearRegression()
            #model_left_log = SGDClassifier(loss='log')
            #model_right_log = SGDClassifier(loss='log')

            #fitting the regression model with a constant
            constant = np.ones((num_samples, 1))
            model_left.fit(np.concatenate((constant, x_inverse_left),axis=1), predictions, sample_weight=weights_left)
            model_right.fit(np.concatenate((constant, x_inverse_right),axis=1), predictions, sample_weight=weights_right)
            #model_left_log.fit(np.concatenate((constant, x_inverse_left),axis=1), predictions, sample_weight=weights_left)
            #model_right_log.fit(np.concatenate((constant, x_inverse_right),axis=1), predictions, sample_weight=weights_right)

            #words:
            left_words.append(['constant'] + f.get_String_Sentence(x_lime_left[0]))
            right_words.append(['constant'] + f.get_String_Sentence(x_lime_right[0]))
            targets.append(f.get_String_Sentence(target_word[index]))

            #coeffs
            left_coefs.append(model_left.coef_)
            right_coefs.append(model_right.coef_)
            #left_coefs.append(model_left_log.coef_)
            #right_coefs.append(model_right_log.coef_)

            results.write('Instance ' + str(index) +':' +'\n')
            results.write('True Label: ' + str(true_label[index]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('Left coefs: ' + str(left_coefs[index]) + '\n')
            results.write('Left words: ' + str(left_words[index]) + '\n')
            results.write('Right coefs: ' + str(right_coefs[index]) + '\n')
            results.write('Right words: ' + str(right_words[index]) + '\n')
            results.write('Target words: ' + str(targets[index]) + '\n')
            results.write('\n')
        results.close()
    end = time.time()
    print("It took: " + str(end - start) + 'seconds')
    print('size: ' + str(size))




    '''
    print(predictions)
    print(probabilities)
    print(np.sum(predictions))
    neg, neu, pos = get_polarityStats(predictions)
    print('sample_size: ' + str(num_samples) +', neg: ' + str(neg)+', neu: ' + str(neu)+', pos: ' + str(pos))
    print('neg: ' + str(neg/num_samples))
    print('neu: ' + str(neu / num_samples))
    print('pos: ' + str(pos / num_samples))
    print('true label: ' + str(polarity[index]))
    '''
    print(f.get_String_Sentence(x_lime_left[0]))
    print(f.get_String_Sentence(x_lime_right[0]))





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
        return np.zeros((num_samples,1)).astype(int)\
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


