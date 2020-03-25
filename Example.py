from classifier import *
from loadData import getStatsFromFile
import numpy as np

#initialisation of object for year 2016:

input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
model_path = 'trainedModelOlaf/2016/-18800'
year = 2016
#initialisation of object for year 2015:
#input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
#model_path = 'trainedModelOlaf/2015/-12800'


A = classifier(input_file, model_path, year) #A is our classifier

# Example 1: getting accuracy of the remaining test data

size, polarity = getStatsFromFile(input_file) #polarity is a vector with the classifications of the instances
predictions = np.array([])
correct = np.array([])
probabilities = np.zeros((int(size),3))
for i in range(int(size)):

    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = A.get_instance(i)
    pred, prob = A.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
    predictions = np.append(predictions, pred)
    probabilities[i,:] = prob
    if(pred == int(polarity[i])):
        correct = np.append(correct,1)
print(probabilities)
print('polarities: ')
print(polarity)
print('predictions: ')
print(predictions)

acc = sum(correct)/size
print('size of sample is: ' + str(size) + ', correct: ' + str(sum(correct)))
print('accuracy: ' + str(acc))

#Example 2: getting word embedding representation of words We don't want to remove the aspect in our perturbations
#so only consider x_left and x_right for our sentences first
''' 
sentence_embedding = A.get_GloVe_embedding(A.x_left[1], len=A.x_left_len[1]) #embedding for 1'th instance
print('embedding of the sentence is given by: ')
print(sentence_embedding)
print(sentence_embedding.shape)
print(A.w2v)
print(A.word_id_mapping)

#Example 3: Getting the words as string from the id's

words = get_Allwords(A.word_id_mapping)

s = A.get_String_Sentence(A.x_left[1])
print(A.x_left.shape)
print(s)

#Example 4: getting all the probabilities
predictions, probabilities = A.get_allProb(A.x_left, A.x_left_len, A.x_right, A.x_right_len, A.y_true, A.target_word, A.target_words_len, 5,int(size))
print(predictions)
print(probabilities)

print(polarity)
'''



