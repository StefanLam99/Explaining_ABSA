from utils import *
from config import *
from lcrModelAlt import *
import tensorflow as tf
from loadData import getStatsFromFile
class classifier:

    def __init__(self, input_file, model_path, year):
        """
        Constructor to initialize a black box model
        :param input_file: he file containing the .txt data of the instances (reviews)
        :param model_path: the path to the trained model
        """
        self.input_file = input_file
        self.year = year
        self.word_id_mapping, self.w2v = load_w2v("data/programGeneratedData/"+str(FLAGS.embedding_dim)+'embedding'+str(self.year)+".txt", FLAGS.embedding_dim)

        self.x_left, self.x_left_len, self.x_right, self.x_right_len, self.y_true, self.target_word, \
        self.target_words_len, _, _, _ = load_inputs_twitter(input_file, self.word_id_mapping, FLAGS.max_sentence_len,
                                                             'TC', FLAGS.is_r == '1', FLAGS.max_target_len)
        ##restoring the trained model
        # delete the current graph
        tf.reset_default_graph()
        # import the loaded graph
        self.imported_graph = tf.train.import_meta_graph(model_path + '.meta')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
            # restore the saved variables
        self.imported_graph.restore(self.sess, model_path)
        self.graph = tf.get_default_graph()

        # setting keys for the feed dict
        self.x = self.graph.get_tensor_by_name('inputs/x:0')
        self.y = self.graph.get_tensor_by_name('inputs/y:0')
        self.sen_len = self.graph.get_tensor_by_name('inputs/sen_len:0')
        self.x_bw = self.graph.get_tensor_by_name('inputs/x_bw:0')
        self.sen_len_bw = self.graph.get_tensor_by_name('inputs/sen_len_bw:0')
        self.target_words = self.graph.get_tensor_by_name('inputs/target_words:0')
        self.tar_len = self.graph.get_tensor_by_name('inputs/tar_len:0')
        self.keep_prob1 = self.graph.get_tensor_by_name('keep_prob1:0')
        self.keep_prob2 = self.graph.get_tensor_by_name('keep_prob2:0')
        self.prob = self.graph.get_tensor_by_name('softmax/prediction:0')


    def get_instance(self, index):
        """
        Method to get one instance at index.
        :param index: this is the index of the instance we get
        :return: the input parameters of one instance (see get_prob for the definion of each parameter)
        """

        return self.x_left[index:index+1], self.x_left_len[index:index+1], self.x_right[index:index+1], \
               self.x_right_len[index:index+1], self.y_true[index:index+1], self.target_word[index:index+1], \
               self.target_words_len[index:index+1]




    def get_prob(self, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len):
        """
        Method to get the probability vector of ONE instance based on the model of...
        Use get_instance to get the respective parameters
        :param x_left: the words to the left of the targets represented by a sequence id's
        :param x_left_len: number of nonzeros in x_left, i.e., number of words to the left of the target
        :param x_right: the words to the right of the targets represented by a sequence id's
        :param x_right_len: number of nonzeros in x_left, i.e., number of words to the right of the target
        :param y_true: true label of the sentence ( don't use this one)
        :param target_word: the target words in the sentence represented by a sequence of id's
        :param target_words_len: number of nonzeros in target_words, i.e., number of words to the right of the target
        :param model_path: the path to the trained model of...
        :return: a probability vector that classifies the instance
        """


        #add an useless instance to the dict to make the program run, we will remove it later.
        feed_dict = {
                self.x: np.concatenate((x_left, np.zeros([1,FLAGS.max_sentence_len]))),
                self.x_bw: np.concatenate((x_right, np.zeros([1,FLAGS.max_sentence_len]))),
                self.y: np.concatenate((y_true, np.zeros([1,FLAGS.n_class]))),
                self.sen_len: np.concatenate((x_left_len,np.zeros(1))),
                self.sen_len_bw: np.concatenate((x_right_len,np.zeros(1))),
                self.target_words: np.concatenate((target_word, np.zeros([1,FLAGS.max_target_len]))),
                self.tar_len: np.concatenate((target_words_len,np.zeros(1))),
                self.keep_prob1: 1,
                self.keep_prob2: 1,
            }

       ##getting prediction of instance
        prob = self.sess.run(self.prob,feed_dict=feed_dict)
        prob = orderProb(prob,self.year)
        pred = np.argmax(prob[0]) - 1

        return pred, prob[0] #only get the relevant probability

    def get_Allinstances(self):
        """
        method to return all instances in a dictionary
        :return:
        """
        size, polarity = getStatsFromFile(self.input_file)
        size = int(size)
        correctSize = 0
        predictions, probabilities = self.get_allProb( self.x_left, self.x_left_len, self.x_right, self.x_right_len, self.y_true,
                                                 self.target_word, self.target_words_len, size, size)
        correctDict = {
            'x_left': [],
            'x_left_len': [],
            'x_right': [],
            'x_right_len': [],
            'target': [],
            'target_len': [],
            'y_true': [],
            'true_label': [],
            'pred': []

        }
        for i in range(size):
                correctDict['x_left'].append(self.x_left[i])
                correctDict['x_right'].append(self.x_right[i])
                correctDict['x_left_len'].append(self.x_left_len[i])
                correctDict['x_right_len'].append(self.x_right_len[i])
                correctDict['target'].append(self.target_word[i])
                correctDict['target_len'].append(self.target_words_len[i])
                correctDict['y_true'].append(self.y_true[i])
                correctDict['true_label'].append(int(polarity[i]))
                correctDict['pred'].append(predictions[i])
                correctSize +=1
        correctDict['size'] = correctSize
        return correctDict
    def get_split_instances(self):
        """
        Splits the instances in correct and incorrect as dictionaries
        :return:
        """
        size, polarity = getStatsFromFile(self.input_file)
        size = int(size)
        predictions, probabilities = self.get_allProb( self.x_left, self.x_left_len, self.x_right, self.x_right_len, self.y_true,
                                                 self.target_word, self.target_words_len, size, size)
        correctDict = {
            'x_left': [],
            'x_left_len': [],
            'x_right': [],
            'x_right_len': [],
            'target': [],
            'target_len': [],
            'y_true': [],
            'true_label': [],
            'pred': []

        }

        incorrectDict = {
            'x_left': [],
            'x_left_len': [],
            'x_right': [],
            'x_right_len': [],
            'target': [],
            'target_len': [],
            'y_true': [],
            'true_label': [],
            'pred': []
        }
        correctSize = 0
        incorrectSize = 0
        for i in range(size):
            if(int(polarity[i]) == int(predictions[i])):
                correctDict['x_left'].append(self.x_left[i])
                correctDict['x_right'].append(self.x_right[i])
                correctDict['x_left_len'].append(self.x_left_len[i])
                correctDict['x_right_len'].append(self.x_right_len[i])
                correctDict['target'].append(self.target_word[i])
                correctDict['target_len'].append(self.target_words_len[i])
                correctDict['y_true'].append(self.y_true[i])
                correctDict['true_label'].append(int(polarity[i]))
                correctDict['pred'].append(predictions[i])
                correctSize +=1
            else:
                incorrectDict['x_left'].append(self.x_left[i])
                incorrectDict['x_right'].append(self.x_right[i])
                incorrectDict['x_left_len'].append(self.x_left_len[i])
                incorrectDict['x_right_len'].append(self.x_right_len[i])
                incorrectDict['target'].append(self.target_word[i])
                incorrectDict['target_len'].append(self.target_words_len[i])
                incorrectDict['y_true'].append(self.y_true[i])
                incorrectDict['true_label'].append(int(polarity[i]))
                incorrectDict['pred'].append(predictions[i])
                incorrectSize +=1

        correctDict['size'] = correctSize
        incorrectDict['size'] = incorrectSize
        return correctDict, incorrectDict

    def get_allProb(self, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, batch_size,num_samples):
        """
        Almost the same as get_prob, but here we input all instances at the same time and get a probability matrix
        and prediction vector. Input are arrays/matrices with as row length the sample size.
        """
        probabilities = np.zeros((num_samples,FLAGS.n_class))
        predictions = np.zeros(num_samples)
        for i in range(int(num_samples/batch_size)):
            batch_start = i*batch_size
            batch_end = (i+1)*batch_size

            feed_dict = {
                self.x: x_left[batch_start:batch_end],
                self.x_bw: x_right[batch_start:batch_end],
                self.y: y_true[batch_start:batch_end],
                self.sen_len: x_left_len[batch_start:batch_end],
                self.sen_len_bw: x_right_len[batch_start:batch_end],
                self.target_words: target_word[batch_start:batch_end],
                self.tar_len: target_words_len[batch_start:batch_end],
                self.keep_prob1: 1,
                self.keep_prob2: 1,
            }

            ##getting prediction of instance

            prob = self.sess.run(self.prob, feed_dict=feed_dict)
            prob = orderProb(prob, self.year)
            pred = np.argmax(prob, axis=1) - 1
            probabilities[batch_start:batch_end,:] = prob
            predictions[batch_start:batch_end] = pred

        return predictions, probabilities


    def get_GloVe_embedding(self, sentence, len):
        """
        Method to get a word embedding of a sentence with id's as word representations
        :param sentence: array with length 80 (max sentence length)
        :return: the word embedding of each word (dim = (300,80))
        """
        sentence_embedding = np.zeros((FLAGS.embedding_dim, FLAGS.max_sentence_len))

        for i in range(len):
            sentence_embedding[:, i] = self.w2v[sentence[i], :]
            if(int(np.sum(sentence_embedding[:,i])) == 0):
                sentence_embedding[:,i] = np.random.normal(loc=0, scale=0.05**2,size=(300)) #if word doesnt exist

        return sentence_embedding


    def get_String_Sentence(self, sentence):
        """
        Method to get a string representation of a sentence that consists of id's
        :param sentence: words with id representation
        :return: sentence with real words as a string
        """
        words = get_Allwords(self.word_id_mapping)
        s = []
        for i in range(len(sentence)):
            if sentence[i] != 0:
                 s.append(words[sentence[i]-1])

        return s

    def get_all_sentences(self, sentence_matrix):
        nr, nc = sentence_matrix.shape
        words = get_Allwords(self.word_id_mapping)
        s = []
        for i in range(nr):
            sentence = []
            for j in range(nc):
                if sentence_matrix[i][j] != 0:
                    sentence.append(words[sentence_matrix[i][j]-1])
            s.append(sentence)

        return s
