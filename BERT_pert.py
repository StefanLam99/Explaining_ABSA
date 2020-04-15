from spacy.vocab import Vocab
import pandas as pd
import en_core_web_lg
from numpy import unicode
import numpy as np
import sys
import time
from classifier import *
from spacy.tokenizer import Tokenizer
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

class Neighbors:
    def __init__(self, nlp_obj):
        file = 'data/programGeneratedData/768embedding2016.txt'
        df = pd.read_csv(file, sep=" ", encoding='cp1252', header=None)
        df = df.drop(columns=769)
        D = {}  # dictionary of all words and vectors in bert semeval data
        L = df.loc[:, 0].values  # list of all words
        for i, word in enumerate(L):
            D[word] = df.loc[i, 1:].values

        self.vocab = Vocab()
        for word, vector in D.items():
            self.vocab.set_vector(word, vector)
        self.nlp = nlp_obj
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        self.to_check = [self.vocab[w] for w in self.vocab.strings]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.vocab.strings:
                self.n[word] = []
            else:
                word = self.vocab[unicode(word)]
                queries = [w for w in self.to_check]

                by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(by_similarity[0].orth_)[0], word.similarity(by_similarity[0]))]
                self.n[orig_word] += [(self.nlp(w.orth_)[0], word.similarity(w))
                                 for w in by_similarity[100:600] if self.nlp(word.orth_)[0].text.split('_')[0] != self.nlp(w.orth_)[0].text.split('_')[0]]


        return self.n[orig_word]



def perturb_sentence(text, n, neighbors, proba_change=0.5,
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
    normal_text = [w.split('_')[0] for w in text]
    nomrla_text = ' '.join(normal_text)
    normal_tokens = neighbors.nlp(unicode(normal_text))
    bert_text = ' '.join(text)
    bert_tokens = neighbors.nlp(unicode(bert_text))
    # print [x.pos_ for x in tokens]
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(bert_tokens)), '|S80')
    data = np.ones((n, len(bert_tokens)))
    raw[:] = [x.text for x in bert_tokens]
    for i, t in enumerate(normal_tokens):
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(bert_tokens[i].text)
                if neighbors.nlp(x[0].text.split('_')[0])[0].tag_ == t.tag_][:top_n]
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
                print(t_neighbors)
                raw[:, i] = np.random.choice(t_neighbors, n, p = weights,
                                             replace=True)
                print(raw[:,i])
                print('hello')
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



def get_perturbations(pert_left, pert_right, neighbors, b, i, num_samples):
    # pert_left is a boolean if the left part has to be perturbed
    # i is the index of the instance that has to be perturbed

    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = b.get_instance(i)
    instance_sentiment, prob = b.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len)
    x = [x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len]
    if pert_left:
        text = b.get_String_Sentence(b.x_left[i])
        text = [w.replace('–', '-') for w in text]
        print(text)

        #text = [s.encode() for s in text]
    if pert_right:
        text = b.get_String_Sentence(b.x_right[i])
        #text = x_right_sentence[::-1]
        text = [w.replace('–', '-') for w in text]
        print(text)

    #nlp = en_core_web_lg.load()
    present = []
    #neighbors = Neighbors(nlp)


    raw_data, data = perturb_sentence(text, num_samples, neighbors, proba_change=0.5,
                                      top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                                      forbidden_words=['be'],
                                      pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=False,
                                      temperature=.4)
    perturbations = []
    output_data = []
    for i in range(0, len(raw_data)):
        new_data = raw_data[i].replace('"', "'")
        '''  
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
        '''
        output_data.append(new_data)
    perturbations = output_data

    return perturbations, instance_sentiment, text, b, x

'''' 
if __name__ == '__main__':
    begin = time.time()
    f = classifier('Maria')
    index = 4
    num_samples = 5000
    nlp = en_core_web_lg.load()
    neighbors = Neighbors(nlp)
    print(neighbors.neighbors('nice'))
    print(neighbors.neighbors('nice_81'))
    print(len(neighbors.neighbors('nice_81')))

    perturbationsl, instance_sentiment, text, b, x = get_perturbations(False, True, neighbors, f, index, num_samples)
    perturbationsr, instance_sentiment, text, b, x = get_perturbations(True, False, neighbors, f, index, num_samples)

    print(f.sentence_at(index))
    print(perturbationsl)
    print(perturbationsr)
    print(time.time() -begin)

'''