from classifier import *
from loadData import getStatsFromFile
import numpy as np
input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
model_path = 'trainedModel/2016/-2778'
size, polarity = getStatsFromFile(input_file)

for i in range(int(size)):
    counter = 0
    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = get_instance(i, input_file)
    pred, prob = get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, model_path)
    if(pred == polarity[i]):
        counter += 1

acc = size/counter
print(prob)
print(size)
print(polarity)
print('accuracy: ' + str(acc))

