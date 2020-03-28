from __future__ import print_function
from Lime import lime_perturbation
from classifier import *
from sklearn.utils import check_random_state


class Decision:
    def __init__(self, c, word):
        self.c = c
        self.word = word
        self.flag = None

    def match(self, example):
        """
        method to match different instances/observation with eachother, for a feature
        :param example: another instance/obervation
        :return: true if it matches, false else
        """
        word = example[self.c]

        if(self.word == word):
            self.flag = True
        else:
            self.flag = False

        return self.flag

    def __repr__(self):
            return "Is %f inside instance x?" %(self.word)

def split(rows, decision):
    true_rows, false_rows = [], []
    for row in rows:
        if decision.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_decision = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)

    n_features = len(rows[0]) -1 # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value
            decision = Decision(col, val)

            # try splitting the dataset
            true_rows, false_rows = split(rows, decision)
            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_decision = gain, decision

    return best_gain, best_decision

class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 decision,
                 true_branch,
                 false_branch):
        self.decision = decision
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    gain, decision = find_best_split(rows)
    print(gain)
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = split(rows, decision)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(decision, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.decision))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.decision.match(row):#modified
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



def data(year, model, case, index):
    """
    Preprocesses data
    :param year:
    :param model:
    :param case:
    :param index:
    :return:
    """


    num_samples = 5000
    batch_size = 500
    r = check_random_state(2020)
    if year == 2015:
        input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
        model_path = 'trainedModelOlaf/2015/-12800'
    elif year == 2016:
        input_file = 'data/programGeneratedData/300remainingtestdata2016.txt'
        model_path = 'trainedModelOlaf/2016/-18800'

    f = classifier(input_file=input_file, model_path=model_path, year=year)
    dict = f.get_Allinstances()
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

    x_inverse_left, x_lime_left, x_lime_left_len = lime_perturbation(r, x_left[index], x_left_len[index], num_samples)
    x_inverse_right, x_lime_right, x_lime_right_len = lime_perturbation(r, x_right[index], x_right_len[index],
                                                                        num_samples)
    target_lime_word = np.tile(target_word[index], (num_samples, 1))
    target_lime_word_len = np.tile(target_words_len[index], (num_samples))
    y_lime_true = np.tile(y_true[index], (num_samples, 1))

    # predicting the perturbations
    predictions, probabilities = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                               y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                               num_samples)

    sentences_left = np.array(f.get_all_sentences(x_lime_left))
    sentences_right = np.array(f.get_all_sentences(x_lime_right))
    predictions = predictions.reshape(num_samples,1)

    return predictions, np.concatenate((x_inverse_left,predictions),axis=1), sentences_left, \
    np.concatenate((x_inverse_right,predictions),axis=1),sentences_right

def main():
    year = 2015
    model = 'test'
    case = 'all'  # correct, incorrect or nothing for all
    index = 250
    predictions, x_inverse_left, left_sentence, x_inverse_right, right_sentence = data(year, model, case, index=index)
    '''
    print(x_inverse_right)
    print(right_sentence)
    dictionary = class_counts(x_inverse_left)
    print(dictionary)
    '''
    print(x_inverse_left)
    print(x_inverse_right)
    my_tree = build_tree(x_inverse_left)
    print_tree(my_tree)

    for row in x_inverse_right:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))

if __name__ == '__main__':
    main()
