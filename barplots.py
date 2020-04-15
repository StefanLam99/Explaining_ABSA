import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


'''
def create_Limeplot(words, influences, type, save_path, predicted, true):
    plt.rcdefaults()

    fig, ax = plt.subplots()

    components = {'Right Context': 'red', 'Left Context': 'blue'}
    labels = list(components.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=components[label]) for label in labels]
    leg=plt.legend(handles, labels, loc='lower right')

    leg.get_frame().set_edgecolor('black')
    plt.axvline(x=0, color='black')
    y_pos = np.arange(len(words))


    ax.barh(y_pos, influences, align='center', color=type)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Influence towards the "' +predicted+ '" Sentiment')
    ax.set_title('True Label: ' + true+ '\n' + 'Predicted Label:  '+ predicted )
    plt.savefig( save_path + '.png', bbox_inches='tight')

    plt.show()


    #Maria wsp
words = ['bad','have','been' ,  'I']
type = ['crimson','darkblue', 'darkblue', 'darkblue', ]
influences = [5.9748,-1.6489,-1.7924,  -1.9733]
save_path = 'data/Lime/plots/WSPmaria'
predicted = 'Negative'
true = 'Negative'
create_Limeplot(words, influences, type, save_path, predicted, true)
#olaf
words = ['bad', 'is', ',', 'it']
type = ['crimson','crimson', 'crimson', 'crimson', ]
influences = [8.612,2.02,0.74, 0.708]
save_path = 'data/Lime/plots/WSPmariaOlaf'
predicted = 'Negative'
true = 'Negative'
create_Limeplot(words, influences, type, save_path, predicted, true)

#Maria SP
words = ['service','customer','/' , 'Poor' ]
type = ['darkblue','darkblue', 'darkblue', 'darkblue', ]
influences = [2.196,-5.56,-5.654,  -5.6737]
influences = [5.6737,5.654,5.56, -2.196 ]
save_path = 'data/Lime/plots/SPmaria'
predicted = 'Negative'
true = 'Negative'
create_Limeplot(words, influences, type, save_path, predicted, true)
#olaf
words = ['Poor','customer','service' ,  '/']
type = ['darkblue','darkblue', 'darkblue', 'darkblue', ]
influences = [9.841,0.1375,0.087,  0.038]
save_path = 'data/Lime/plots/SPmariaOlaf'
predicted = 'Negative'
true = 'Negative'
create_Limeplot(words, influences, type, save_path, predicted, true)

# Maria RP
words = ['I','top','list' ,  'of']
type = ['darkblue','crimson', 'crimson', 'crimson', ]
influences = [2.56, 1.853, -3.3,  -3.65]
save_path = 'data/Lime/plots/RPmaria'
predicted = 'Positive'
true = 'Neutral'
create_Limeplot(words, influences, type, save_path, predicted, true)

# Olaf RP
words = ['list','want','of' ,  'I']
type = ['crimson','darkblue', 'crimson', 'darkblue', ]
influences = [1.335, 1.019, 0.97,  -0.974]
save_path = 'data/Lime/plots/RPolaf'
predicted = 'Negative'
true = 'Neutral'
create_Limeplot(words, influences, type, save_path, predicted, true)

#Olaf SP
words = ['outstanding', 'excellent', 'food', 'by']
type = ['crimson','darkblue', 'crimson', 'Crimson', ]
influences = [2.142, 1.162, -1.1765,  -1.327]
save_path = 'data/Lime/plots/SPolaf'
predicted = 'Positive'
true = 'Positive'
create_Limeplot(words, influences, type, save_path, predicted, true)

#Olaf WSP
words = ['annoying', 'through', 'dinner', 'became']
type = ['crimson','crimson', 'crimson', 'Crimson', ]
influences = [6.6977, 1.23, 1.15,  -1.06]
save_path = 'data/Lime/plots/WSPolaf'
predicted = 'Negative'
true = 'Negative'
create_Limeplot(words, influences, type, save_path, predicted, true)



def create_plots(left_bars, right_bars, words, save_path, title):
    barwidth = 0.9
    n_left = len(left_bars)
    n_right = len(right_bars)
    bars = left_bars + right_bars

    rleft =  list(range(1,n_left+1))
    rright = list(range(n_left+1, n_left+n_right+1))
    rall = rleft + rright
    print(rleft)
    print(rright)

    # Create barplot
    plt.barh(rleft, left_bars, width=barwidth, color=('darkblue'), label='Left Context')
    plt.bar(rright, right_bars, width=barwidth, color=('crimson'), label='Right Context')
    plt.legend()
    plt.yticks([r + barwidth for r in range(len(rall))],
               words, rotation=90)
    plt.title('True Label: ' + str(title[0]) + '\n' + 'Predicted Label: ' + str(title[1]))
    plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.gca().invert_xaxis()
    plt.savefig(save_path + '.png', bbox_inches='tight')
    plt.show()

left_bars = [-1.9733, -1.6489, -1.7924]
right_bars = [5.9748]
words = ['I', 'have', 'been', 'bad']
title = ['Negative', 'Negative']
save_path = 'test'
create_plots(left_bars, right_bars, words, save_path, title)
'''
def Barplots(negBar, NeuBar, posBar,words, save_path, true, predicted):
    # set width of bar
    barWidth = 0.25


    # Set position of bar on X axis
    r1 = np.arange(len(negBar))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, neuBar, color='#7f6d5f', width=barWidth, edgecolor='white', label='Neutral Sentiment')
    plt.bar(r2, negBar, color='#2d7f5e', width=barWidth, edgecolor='white', label='Negative Sentiment')
    plt.bar(r3, posBar, color='#557f2d', width=barWidth, edgecolor='white', label='Positive Sentiment')

    # Add xticks on the middle of the group bars
    plt.ylabel('Influences', fontweight='bold', fontsize= 'medium')
    plt.yticks(fontsize=12)
    plt.xticks([r + barWidth for r in range(len(posBar))], words, fontsize=14)
    plt.axhline(y=0, color='black', linestyle='solid')
    plt.title('True Label: ' + true + '\n' + 'Predicted Label: ' + predicted)
    plt.legend(title="Influence towards:", loc=2, fancybox=True, edgecolor = 'black')
    # Create legend & Show graphic
    plt.savefig(save_path + '.png', bbox_inches='tight')
    plt.show()

#SP OLAF
#['the']['food']['is', 'fantastic', ',', 'and', 'the', 'waiting', 'staff', 'has', 'been', 'perfect', 'every', 'single', 'time', 'we', "'ve", 'been', 'there', '.']

negBar = [-0.65878372, -0.45700045, -0.77838813, -0.85068953]
neuBar = [ 0.13960276,  0.28916272, -0.48187059, -0.39411159]
posBar = [0.4050801,  0.14247899, 0.96302864, 1.06382518]
words = ['every', 'the', 'perfect', 'fantastic']
save_path = 'data/Lime2/plots/SPolaf'
true = 'Positive'
predicted = 'Positive'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#WSP OLAF
#['the']['appetizer']['was', 'interesting', ',', 'but', 'the', 'creme', 'brulee', 'was', 'very', 'savory', 'and', 'delicious', '.']

negBar = [-0.55678208, -2.8318922,  -3.0224082,  -1.02272379]
neuBar = [-2.64604238e-10, -1.81849716e-10, -3.10431311e-10, -6.93506486e-11]
posBar = [0.55678208, 2.8318922,  3.0224082,  1.02272379]
words = ['creme', 'interesting', 'delicious', 'very']
save_path = 'data/Lime2/plots/WSPolaf'
true = 'Positive'
predicted = 'Positive'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)

#RP OLAF
#['the', 'only', 'positive', 'thing', 'about']['mioposto']['is', 'the', 'nice', 'location', '.']
negBar = [-0.52116307, -0.44910487, -1.64872598, -1.30323279]
neuBar = [-0.37060114, -0.67028783, -0.29629573, -0.46953773]
posBar = [0.59295018, 0.49955694, 1.70458201, 1.33495427]
words = ['thing', 'the', 'nice', 'location']
save_path = 'data/Lime2/plots/RPolaf'
true = 'Negative'
predicted = 'Positive'
Barplots(negBar, neuBar, posBar,words, save_path, true, predicted)
