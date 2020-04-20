import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

def Barplots(negBar, NeuBar, posBar,words, save_path, true, predicted):
    # set width of bar
    barWidth = 0.25


    # Set position of bar on X axis
    r1 = np.arange(len(negBar))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, neuBar, color='#3ac0da', width=barWidth, edgecolor='white', label='Neutral Sentiment')
    plt.bar(r2, negBar, color='#01579b', width=barWidth, edgecolor='white', label='Negative Sentiment')
    plt.bar(r3, posBar, color='#78FFFF', width=barWidth, edgecolor='white', label='Positive Sentiment')

    # Add xticks on the middle of the group bars
    plt.ylabel('Influences', fontweight='bold', fontsize= 'medium')
    plt.yticks(fontsize=12)
    plt.xticks([r + barWidth for r in range(len(posBar))], words, fontsize=14)
    plt.axhline(y=0, color='black', linestyle='solid')
    plt.title('Prediction differences of instance: ' + true + ' for the ' + predicted)
    plt.legend(title="Influence towards:", loc=2, fancybox=True, edgecolor = 'black')
    # Create legend & Show graphic
    plt.savefig(save_path + '.png', bbox_inches='tight')
    plt.show()

#SP OLAF
#['the']['food']['is', 'fantastic', ',', 'and', 'the', 'waiting', 'staff', 'has', 'been', 'perfect', 'every', 'single', 'time', 'we', "'ve", 'been', 'there', '.']

negBar = [-0.00109057 ]
neuBar = [ -0.0001973]
posBar = [0.00128782]
words = ['location']
save_path = 'data/LACE/plots/152LORE'
true = '152'
predicted = 'LORE model'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#WSP OLAF
#['the']['appetizer']['was', 'interesting', ',', 'but', 'the', 'creme', 'brulee', 'was', 'very', 'savory', 'and', 'delicious', '.']

negBar = [-1.08866971e-07]
neuBar = [2.31128752e-08]
posBar = [1.19209290e-07]
words = ["'ve"]
save_path = 'data/LACE/plots/154LORE'
true = '154'
predicted = 'LORE model'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#RP OLAF
#['the', 'only', 'positive', 'thing', 'about']['mioposto']['is', 'the', 'nice', 'location', '.']
negBar = [-0.02597801, -1.04290248e-05, -7.30426291e-06]
neuBar = [-0.00010079  , -1.82397471e-07, -5.14377660e-08]
posBar = [0.02607876, 1.06096268e-05, 7.27176666e-06]
words = ['delicious', 'savory', 'very']
save_path = 'data/LACE/plots/275LORE'
true = '275'
predicted = 'LORE model'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

negBar = [2.01983163e-04, -0.01125765, -8.21607537e-05, -3.36093944e-04, -0.00109057 ]
neuBar = [ 4.52743479e-05 ,  -0.00049545, -4.49759318e-05, -3.96694886e-05, -0.0001973]
posBar = [-2.47240067e-04,  0.01175314, 1.27136707e-04, 3.75866890e-04, 0.00128782]
words = ['thing', 'nice', 'the', 'is', 'location']
save_path = 'data/LACE/plots/152'
true = '152'
predicted = 'rule based models'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#WSP OLAF
#['the']['appetizer']['was', 'interesting', ',', 'but', 'the', 'creme', 'brulee', 'was', 'very', 'savory', 'and', 'delicious', '.']

negBar = [2.40263489e-07, 1.23946620e-07, -9.10089284e-07, 1.14876809e-06, -1.08866971e-07]
neuBar = [1.60030339e-07, -6.22113703e-08, -5.72679255e-08, 3.27009516e-07, 2.31128752e-08]
posBar = [-3.57627869e-07, 0.00000000e+00, 1.07288361e-06, -1.43051147e-06, 1.19209290e-07]
words = ['waiting', 'has', 'been', ',', "'ve"]
save_path = 'data/LACE/plots/154'
true = '154'
predicted = 'rule based models'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#RP OLAF
#['the', 'only', 'positive', 'thing', 'about']['mioposto']['is', 'the', 'nice', 'location', '.']
negBar = [-7.30426291e-06,  8.54646089e-06 , 5.81028326e-06, -0.02597801, -1.04290248e-05]
neuBar = [-5.14377660e-08, 8.05208114e-08,  5.81134110e-08, -0.00010079  , -1.82397471e-07]
posBar = [7.27176666e-06, -8.70227814e-06,  -5.96046448e-06, 0.02607876, 1.06096268e-05]
words = ['very', 'was', 'but', 'delicious', 'savory']

# 'the', -1.50353208e-05 -3.70028843e-07,1.53779984e-05,
save_path = 'data/LACE/plots/275'
true = '275'
predicted = 'rule based models'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

