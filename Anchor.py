
from classifier import *
from config import *
import nltk
import sys
from numpy import unicode
import numpy as np
np.random.seed(2020)
#from np.magic import np
import en_core_web_lg
import en_core_web_sm
import os
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

'''
class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            idx = nltk.text.ContextIndex([word.lower() for word in nltk.corpus.brown.words()])
            queries = []
            for word in nltk.word_tokenize(word):
                queries.append(idx.similar_words(word))
            saveNew = []
            for i in range(0, len(queries[0])):
                saveNew.append(self.nlp.vocab[unicode(queries[0][i])])
            word = self.nlp.vocab[unicode(word)]
            by_similarity = sorted(saveNew, key=lambda w: word.similarity(w), reverse=True)
            self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                 for w in by_similarity[:500]]
        return self.n[orig_word]


def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True,
                     temperature=.4):
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(unicode(text))
    # print([x.pos_ for x in tokens])
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                              (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                              for x in neighbors.neighbors(t.text)
                              if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                raw[:, i] = np.random.choice(t_neighbors, n, p=weights,
                                             replace=True)
                # The type of data in raw is byte.
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                for j in range(len(weights)):
                    if weights[j] < 0:
                        weights[j] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
    #         else:
    #             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    # print raw
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data
'''
class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []
            else:
                word = self.nlp.vocab[unicode(word)]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
                                    #  if w.lower_ != word.lower_]
        return self.n[orig_word]

def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True,
                     temperature=.4):
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(unicode(text))
    # print [x.pos_ for x in tokens]
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0].encode('utf-8', errors='ignore') for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                for j in range(len(weights)):
                    if weights[j] < 0:
                        weights[j] = 0
                # print t.text
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                raw[:, i] = np.random.choice(t_neighbors, n,  p=weights,
                                             replace=True)
                data[:, i] = raw[:, i] == t.text
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                for j in range(len(weights)):
                    if weights[j] < 0:
                        weights[j] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
#         else:
#             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    # print raw
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data

def get_perturbations(pert_left, pert_right, b, i):
    # pert_left is a boolean if the left part has to be perturbed
    # i is the index of the instance that has to be perturbed

    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = b.get_instance(i)
    instance_sentiment, prob = b.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
    x = [x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len]
    if pert_left:
        text = b.get_String_Sentence(b.x_left[i])
        text = [w.replace('–', '-') for w in text]

        #text = [s.encode() for s in text]
    if pert_right:
        text = b.get_String_Sentence(b.x_right[i])
        #text = x_right_sentence[::-1]
        text = [w.replace('–', '-') for w in text]

    nlp = en_core_web_lg.load()
    present = []
    neighbors = Neighbors(nlp)
    num_samples = 2000
    print(text)
    raw_data, data = perturb_sentence(text, present, num_samples, neighbors, proba_change=0.5,
                                      top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                                      forbidden_words=['be'],
                                      pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=False,
                                      temperature=.4)
    perturbations = []
    output_data = []
    for i in range(0, len(raw_data)):
        new_data = raw_data[i].replace('"', "'")
        new_data = new_data.replace(" ' , ' ", " ")
        new_data = new_data.replace(" '", "")
        new_data = new_data.replace("' ", "")
        new_data = new_data.replace("[", "")
        new_data = new_data.replace("]", "")
        new_data = new_data.replace(" ve ", " 've ")
        new_data = new_data.replace(" s ", " 's ")
        new_data = new_data.replace(" re ", " 're ")
        new_data = new_data.replace(" m ", " 'm ")
        new_data = new_data.replace(" ll ", " 'll ")
        new_data = new_data.replace(" d ", " 'd ")
        new_data = new_data.replace(" ino ", " 'ino ")
        output_data.append(new_data)
    perturbations = output_data
    print(raw_data)

    return perturbations, instance_sentiment, text, b, x


def same_anchors(anchor_list, anchor_new):
    counter = 0
    for x in range(len(anchor_list)):
        if sorted(anchor_list[x]) == sorted(anchor_new):
            counter += 1
    if counter == 0:
        return True
    else:
        return False

def get_coverage(anchor, perturbed_instances):
    # method returns the coverage of an anchor
    # anchor is a list of all anchor words
    # perturbed_instances is a list where each element is a string with the perturbed instance
    # problems: 1. case sensitive and 2. word and word./word!/word? are not the same...
    coverage = 0
    divisor = len(perturbed_instances)
    for x in perturbed_instances:
        sentence = x
        counter = 0
        for y in anchor:
            if y in sentence:
                counter = counter + 1
        if counter == len(anchor):
            coverage += 1
    total_coverage = coverage/divisor
    # print('total coverage',anchor ,total_coverage)
    return total_coverage


def possible_anchor(previous_anchor, instance, minimal_coverage, perturbed_instances):
    # method creates new candidate anchors (in list form) and returns these as a list
    # previousAnchor is a list of the previous anchor words
    # instance is the sentence, a string
    # perturbed_instances is a list of all the perturbed instances as strings
    # minimalCoverage is the coverage that we desire, a decimal number
    instance_list = instance
    good_anchor_list = []
    if previous_anchor == []:
        for x in instance_list:
            if (x != '.'):
                new_anchor = previous_anchor
                if x not in previous_anchor:
                    new_anchor = new_anchor + [x]
                    if get_coverage(new_anchor, perturbed_instances) > minimal_coverage:
                        if same_anchors(good_anchor_list, new_anchor):
                            good_anchor_list.append(new_anchor)
    for y in previous_anchor:
        for x in instance_list:
            if (x != '.'):
                new_anchor = y
                if x not in y:
                    new_anchor = new_anchor + [x]
                    if get_coverage(new_anchor, perturbed_instances) > minimal_coverage:
                        if same_anchors(good_anchor_list, new_anchor):
                            good_anchor_list.append(new_anchor)
    return good_anchor_list


def conditional_distribution(anchor, perturbed_instances_left, perturbed_instances_right):
    # anchor is a list of the anchor words
    # perturbed_instances is a list
    conditional_instances = []
    perturbed_instances = [''] * len(perturbed_instances_right)

    for i in range(len(perturbed_instances_left)):
        perturbed_instances[i] = perturbed_instances_left[i] + perturbed_instances_right[i]
    for x in perturbed_instances:
        counter = 0
        for y in anchor:
            if y in x:
                counter += 1
        if counter == len(anchor):
            conditional_instances.append(x)
    return conditional_instances


def kl_bernoulli(p, q):
    p = min(0.9999999999999999, max(0.0000001, p))
    q = min(0.9999999999999999, max(0.0000001, q))
    return (p * np.log(float(p) / q) + (1 - p) *
            np.log(float(1 - p) / (1 - q)))


def dup_bernoulli(p, level):
    # p = mean = ratio between same guesses and maximal possible correct guesses
    # level = beta / n_samples, beta = coefficient, n_samples is how many times we checked
    lm = p
    um = min(min(1, p + np.sqrt(level / 2.)), 1)
    for j in range(1, 17):
        qm = (um + lm) / 2.
#         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
    return um


def dlow_bernoulli(p, level):
    # p = mean = ratio between same guesses and maximal possible correct guesses
    # level = beta  / n_samples, beta = coefficient from compute_beta, n_samples is how many samples we used for this.
    um = p
    lm = max(min(1, p - np.sqrt(level / 2.)), 0)
    for j in range(1, 17):
        qm = (um + lm) / 2.
#        print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
    return lm


def compute_beta(n_features, t, delta):
    # n_features = length of the amount of anchors that we check.
    # t = which round we are in, in while loop is added +1 every time
    # delta = statistical parameter.
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (t ** alpha) / delta)
    return temp + np.log(temp)


def initial_calculations(possible_anchors_list, perturbed_instances_left, perturbed_instances_right, pert_left, pert_right, initial_value, instance_sentiment, delta, f,
                         x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len):
    # initial_stats = the amount of perturbations we wish to run for initialisation
    # perturbed_instances = all perturbations of the instance
    # pert_x = boolean if the perturbed instances are left or right
    # initial_value is the amount of instances we initialize
    # instance_sentiment = sentiment of the instance (pos/neu/neg)
    # delta = statistical parameter.
    # we return a matrix with the means,lower bounds, upper bounds and corresponding candidate anchors and amount
    # of perturbed instances run
    # x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len information about the normal instance
    #print('pos anch', possible_anchors_list, type(possible_anchors_list))
    amount_of_anchors = len(possible_anchors_list)
    initial_matrix =[ [0 for i in range(5)] for j in range(amount_of_anchors)]
    for x in range(amount_of_anchors):
        counter = 0
        predictions = []
        conditional_distribution_left = conditional_distribution(possible_anchors_list[x], perturbed_instances_left, perturbed_instances_right)
        conditional_distribution_right = conditional_distribution(possible_anchors_list[x], perturbed_instances_left, perturbed_instances_right)
        for y in range(initial_value):
            conditional_element_left = conditional_distribution_left[y]
            conditional_element_right = conditional_distribution_right[y]
            #print('cond element', conditional_element, type(conditional_element))
            if pert_left:
                x_left_word = conditional_element_left.split()
                x_left = get_word_id(x_left_word)
            if pert_right:
                x_right_word = conditional_element_right.split()
                x_right = get_word_id((x_right_word))
            pred, prob = f.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
            predictions.append(pred)
            conditional_sentiment = pred
            if conditional_sentiment == instance_sentiment:
                counter += 1
        mean = counter/initial_value
        #print('pos anch', possible_anchors_list[x])
        #print('predictions', predictions, )
        #print('mean', mean, type(mean))
        beta = compute_beta(len(possible_anchors_list[x]), initial_value, delta)
        level = beta/initial_value
        lb = dlow_bernoulli(mean, level)
        #print('lb', lb, type(lb))
        ub = dup_bernoulli(mean, level)
        #print('ub', ub, type(ub))
        initial_matrix[x] = [mean, lb, ub, possible_anchors_list[x], initial_value]
        #print('initial', initial_matrix, type(initial_matrix))
    return initial_matrix


def get_upper_bounds(initial_matrix):
    # returns a matrix with the upper bound in the first colum, then anchor name and then initial_value
    # this function might not be needed
    matrix = [ [ 0 for i in range(3) ] for j in range(len(initial_matrix)) ]
    for v in range(len(initial_matrix)):
        for w in range(3):
            matrix[v][w] = initial_matrix[v][w + 2]
    return matrix


def get_counter_vector(matrix):
    # 4 is the place where lb is in the matrix
    y = [[0 for i in range(1)] for j in range(len(matrix))]
    for i in range(len(matrix)):
        y[i] = matrix[i][4]
    return y


def get_b_best(batch_size, matrix):
    # important for input to be sorted before hand
    # returns a matrix that in has the best values in the first row and the worst in the last row.
    matrix1 = [[0 for i in range(5)] for j in range(batch_size)]
    for i in range(batch_size):
        for j in range(5):
            matrix1[i][j] = matrix[len(matrix) - i - 1][j]

    return matrix1


def get_not_b_best(batch_size, matrix):
    # returns a matrix that in has the other values
    matrix1 = [[0 for i in range(5)] for j in range(len(matrix)-batch_size)]
    for i in range(len(matrix)-batch_size):
        for j in range(5):
            matrix1[i][j] = matrix[i][j]
    return matrix1


def get_mean_vector(matrix):
    # 0 is the place where lb is in the matrix
    y = [[0 for i in range(1)] for j in range(len(matrix))]
    for i in range(len(matrix)):
        y[i] = matrix[i][0]
    return y


def get_lb_vector(matrix):
    # 1 is the place where lb is in the matrix
    y = [[0 for i in range(1)] for j in range(len(matrix))]
    for i in range(len(matrix)):
        y[i] = matrix[i][1]
    return y


def get_ub_vector(matrix):
    # 2 is the place where ub is in the matrix
    y = [[0 for i in range(1)] for j in range(len(matrix))]
    for i in range(len(matrix)):
        y[i] = matrix[i][2]
    return y


def get_ub_index(matrix,batch_size, possible_anchors_list):
    # returns the index of the highest element
    matrix_of_shit = get_not_b_best(batch_size, matrix)
    ub_vector = get_ub_vector(matrix_of_shit)
    hello = max(ub_vector)
    counter = 0
    for i in ub_vector:
        if i == hello:
            break
        else:
            counter += 1
    return counter


def get_word_id(word_sentence):
    word_id_file, classifier.w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    ids = np.zeros([1, FLAGS.max_sentence_len])
    for i, word in enumerate(word_sentence):
        if word in word_to_id:
            ids[0][i] = word_to_id[word]
    return ids


def bbest_anchors(batch_size, possible_anchors_list, epsilon, delta, perturbed_instances_left, perturbed_instances_right, instance_sentiment, initial_value, pert_left, pert_right, b,
                  x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len):
    # batch_size = how many B best anchors we want to keep
    # possible_anchors_list = candidate anchors where we must select the B best from
    # epsilon = tolerance parameter between 0 and 1
    # delta = significance parameter = 0.05
    # perturbed_instances = all perturbations of the instance
    # instance_sentiment = the sentiment of our instance
    # initial_value = amount of instances we initialize
    amount_of_anchors = len(possible_anchors_list)
    initial_matrix = initial_calculations(possible_anchors_list, perturbed_instances_left, perturbed_instances_right, pert_left, pert_right, initial_value, instance_sentiment, delta, b,
                         x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
    initial_matrix
    if len(possible_anchors_list) <= batch_size:
        return initial_matrix
    upper_matrix = get_upper_bounds(initial_matrix)
    sorted_matrix = sorted(initial_matrix)
    # might not use sorted_upper matrix
    sorted_upper = sorted(upper_matrix)
    statistical_sign = False

    bbest_matrix = get_b_best(batch_size, sorted_matrix)
    # boolean value that stops the while loop when we have statistical significance of selecting the B best.
    while statistical_sign == False and max(get_counter_vector(bbest_matrix)) < 100:
        #DO NOT FORGET TO UPDATE T EACH ROUND ONLY NEED TO UPDATE SORTED_MATRIX/ REST UPDATED HERE AT THE START!
        t = 1
        bbest_matrix = get_b_best(batch_size, sorted_matrix)
        bbest_lb = get_lb_vector(bbest_matrix)
        lowest_lb = min(bbest_lb)
        #pas zo highest_ub aan na het aanpassen van get_ub_vector method
        highest_ub_index = get_ub_index(sorted_matrix, batch_size, possible_anchors_list)
        highest_ub = sorted_matrix[highest_ub_index][2]
        # NOTE TO SELF: use and update sorted_matrix, sort it again AT THE END
        if lowest_lb < highest_ub - epsilon:
            # BEGIN MET UPDATEN MEAN, DAN HOEVAAK WE HEBBEN BEKEKEN, DAN LB, DAN UB,
            for x in range(batch_size):
                current_amount_of_good_predictions = bbest_matrix[x][0]*bbest_matrix[x][4]
                # update how many times we evaluated this anchor
                bbest_matrix[x][4] += 1
                sorted_matrix[amount_of_anchors - x - 1][4] += 1
                used_instance = conditional_distribution(bbest_matrix[x][3], perturbed_instances_left, perturbed_instances_right)
                if pert_left:
                    x_left_word = used_instance
                    x_left = get_word_id(x_left_word)
                if pert_right:
                    x_right_word = used_instance
                    x_right = get_word_id(x_right_word)
                pred, prob = b.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
                predicted_sentiment = pred

                if predicted_sentiment == instance_sentiment:
                    current_amount_of_good_predictions += 1
                    # calculate the new mean
                    sorted_matrix[amount_of_anchors - x - 1][0] = current_amount_of_good_predictions/ bbest_matrix[x][4]
                else:
                    sorted_matrix[amount_of_anchors - x - 1][0] = current_amount_of_good_predictions/ bbest_matrix[x][4]
                p = sorted_matrix[amount_of_anchors - x - 1][0]
                beta = compute_beta(amount_of_anchors, t, delta)
                level = beta/sorted_matrix[amount_of_anchors - x - 1][4]
                sorted_matrix[amount_of_anchors - x - 1][1] = dlow_bernoulli(p, level)
                sorted_matrix[amount_of_anchors - x - 1][2] = dup_bernoulli(p, level)
            # NOW WE HAVE TO analyse the upperbound and update its means/lb and ub
            # line of text where we predict sentiment of the sorted_matrix[highest_ub_index][4] element in the conditional distribution
            # use conditional_distribution(anchor = sorted_matrix[highest_ub_index][3], perturbed_instances), which returns our
            # list and take the sorted_matrix[highest_ub_index][4] element, call this predicted_sentiment_ub
            ub_instance = conditional_distribution(sorted_matrix[highest_ub_index][3], perturbed_instances_left, perturbed_instances_right)
            if pert_left:
                x_left_word = ub_instance
                x_left = get_word_id(x_left_word)
            if pert_right:
                x_right_word = ub_instance
                x_right = get_word_id(x_right_word)
            pred, prob = b.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
            predicted_sentiment_ub = pred
            current_good_predic_ub = sorted_matrix[highest_ub_index][0]*sorted_matrix[highest_ub_index][4]
            sorted_matrix[highest_ub_index][4] += 1
            if predicted_sentiment_ub == instance_sentiment:
                current_good_predic_ub += 1
            sorted_matrix[highest_ub_index][0] = current_good_predic_ub/ bbest_matrix[x][4]
            j = sorted_matrix[highest_ub_index][0]
            beta_up = compute_beta(amount_of_anchors, t, delta)
            level_up = beta_up/sorted_matrix[highest_ub_index][4]
            sorted_matrix[highest_ub_index][1] = dlow_bernoulli(j, level_up)
            sorted_matrix[highest_ub_index][2] = dup_bernoulli(j, level_up)
        # when the lower bounds satisfy, you have to make the function
            print(lowest_lb, highest_ub-epsilon)
        elif lowest_lb >= highest_ub - epsilon:
            statistical_sign = True
        # update time t and the sorted_matrix
        t += 1
        sorted_matrix = sorted(sorted_matrix)
    return bbest_matrix





