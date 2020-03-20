from utils import *
from config import *
from lcrModelAlt import *
import tensorflow as tf

word_id_file, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)

input_file = 'data/programGeneratedData/300remainingtestdata2015.txt'
te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _  = load_inputs_twitter(input_file, word_id_file, FLAGS.max_sentence_len, 'TC', FLAGS.is_r == '1', FLAGS.max_target_len)

def get_instance(index, input_file):
    """
    Method to get one instance at index.
    :param index: this is the index of the instance we get
    :param input_file: the file containing the .txt data of the instances (reviews)
    :return: the input parameters of one instance (see get_prob for the definion of each parameter)
    """
    x_left, x_left_len, x_right, x_right_len, y_true, target_words, target_words_len, _, _, _ = load_inputs_twitter(
        input_file, word_id_file, FLAGS.max_sentence_len, 'TC', FLAGS.is_r == '1', FLAGS.max_target_len)

    return x_left[index:index+1], x_left_len[index:index+1], x_right[index:index+1], x_right_len[index:index+1], \
           y_true[index:index+1], target_words[index:index+1], target_words_len[index:index+1]


def get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, model_path):
    """
    Method to get the probability vector of an instance based on the model of...
    :param x_left: the words to the left of the targets represented by a sequence id's
    :param x_left_len: number of nonzeros in x_left, i.e., number of words to the left of the target
    :param x_right: the words to the right of the targets represented by a sequence id's
    :param x_right_len: number of nonzeros in x_left, i.e., number of words to the right of the target
    :param y_true: true label of the sentence
    :param target_word: the target words in the sentence represented by a sequence of id's
    :param target_words_len: number of nonzeros in target_words, i.e., number of words to the right of the target
    :param model_path: the path to the trained model of...
    :return: a probability vector that classifies the instance
    """

    load_dir = model_path
    load_dir = 'trainedModel/2016lol-d1-0.5d2-0.5b-20r-0.07l2-1e-05sen-80dim-300h-300c-3/-282'
    # delete the current graph
    tf.reset_default_graph()
    # import the loaded graph
    imported_graph = tf.train.import_meta_graph(load_dir + '.meta')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # restore the saved variables
        imported_graph.restore(sess, load_dir )
        graph = tf.get_default_graph()

        #setting keys for the feed dict
        x = graph.get_tensor_by_name('inputs/x:0')
        y = graph.get_tensor_by_name('inputs/y:0')
        sen_len = graph.get_tensor_by_name('inputs/sen_len:0')
        x_bw = graph.get_tensor_by_name('inputs/x_bw:0')
        sen_len_bw = graph.get_tensor_by_name('inputs/sen_len_bw:0')
        target_words = graph.get_tensor_by_name('inputs/target_words:0')
        tar_len = graph.get_tensor_by_name('inputs/tar_len:0')
        keep_prob1 = graph.get_tensor_by_name('keep_prob1:0')
        keep_prob2 = graph.get_tensor_by_name('keep_prob2:0')

        #add an useless instance to the dict to make the program run, we will remove it later.
        feed_dict = {
                x: np.concatenate((x_left, np.zeros([1,FLAGS.max_sentence_len]))),
                x_bw: np.concatenate((x_right, np.zeros([1,FLAGS.max_sentence_len]))),
                y: np.concatenate((y_true, np.zeros([1,FLAGS.n_class]))),
                sen_len: np.concatenate((x_left_len,np.zeros(1))),
                sen_len_bw: np.concatenate((x_right_len,np.zeros(1))),
                target_words: np.concatenate((target_word, np.zeros([1,FLAGS.max_target_len]))),
                tar_len: np.concatenate((target_words_len,np.zeros(1))),
                keep_prob1: 1,
                keep_prob2: 1,
            }

       ##getting prediction of instance
        prob = graph.get_tensor_by_name('softmax/prediction:0')
        prob = sess.run(prob,feed_dict=feed_dict)

    return prob[0] #only get the relevant probability