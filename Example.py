from classifier import *
from loadData import getStatsFromFile
import numpy as np

#initialisation of object for year 2015
input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
model_path = 'trainedModel/2015/-320'
A = classifier(input_file, model_path) #A is our classifier

x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = A.get_instance(5, input_file)
pred, prob = A.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, model_path)

# Example 1: getting accuracy of the remaining test data

size, polarity = getStatsFromFile(input_file) #polarity is a vector with the classifications of the instances
predictions = np.array([])
correct = np.array([])
for i in range(int(size)):

    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = A.get_instance(i, input_file)
    pred, prob = A.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, model_path)
    pred = int(pred)
    predictions = np.append(predictions, pred)

    if(pred == int(polarity[i])):
        correct = np.append(correct,1)

print('polarities: ')
print(polarity)
print('predictions: ')
print(predictions)

acc = sum(correct)/size
print('size of sample is: ' + str(size) + ', correct: ' + str(sum(correct)))
print('accuracy: ' + str(acc))

#Example 2: getting word embedding representation of words We don't want to remove the aspect in our perturbations
#so only consider x_left and x_right for our sentences first

sentence_embedding = A.get_GloVe_embedding(A.x_left[1]) #embedding for 1'th instance
print('embedding of the sentence is given by: ')
print(sentence_embedding)
print(sentence_embedding.shape)





