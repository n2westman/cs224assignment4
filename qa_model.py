from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from pdb import set_trace as t

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim
        self.n_hidden_enc = 5

    def encode(self, question_embeddings, question_lengths, context_embeddings, context_lengths):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with tf.variable_scope("QuestionEncoderBiLSTM"):
            # Forward direction cell
            question_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
            # Backward direction cell
            question_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)

            question_outputs, _, = tf.nn.bidirectional_dynamic_rnn(question_lstm_fw_cell, question_lstm_bw_cell, question_embeddings,
                                                  sequence_length=question_lengths, dtype=tf.float64)

        with tf.variable_scope("AnswerEncoderBiLSTM"):
            # Forward direction cell
            context_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
            # Backward direction cell
            context_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)

            context_outputs, _, = tf.nn.bidirectional_dynamic_rnn(context_lstm_fw_cell, context_lstm_bw_cell, context_embeddings,
                                                  sequence_length=context_lengths, dtype=tf.float64)
        return tf.concat(question_outputs, 2), tf.concat(context_outputs, 2)

class Mixer(object):
    def __init__(self):
            self.n_hidden_mix = 200

    def mix(self, bilstm_encoded_questions, bilstm_encoded_contexts, context_lengths):
        # Compute the attention on each word in the context as a dot product of its contextual embedding and the query
        #questions = tf.Print( questions, [tf.shape(questions)])

        # Dimensionalities:
        # bilstm_encoded_questions: samples x question_words x 2*n_hidden_enc
        # bilstm_encoded_contexts: samples x context_words x 2*n_hidden_enc

        # Dimensionalities: 
        # affinity_matrix_L: samples x question_words x context_words
        # normalized_attention_weights_A_q: samples x question_words x context_words
        # normalized_attention_weights_A_d: samples x context_words x question_words
        # attention_contexts_C_q = samples x 2*n_hidden_enc x question_words
        affinity_matrix_L = tf.matmul(bilstm_encoded_questions, tf.transpose(bilstm_encoded_contexts, perm = [0, 2, 1]))
        normalized_attention_weights_A_q = tf.nn.softmax(affinity_matrix_L)
        normalized_attention_weights_A_d = tf.nn.softmax(tf.transpose(affinity_matrix_L, perm = [0, 2, 1]))
        attention_contexts_C_q = tf.transpose(tf.matmul(normalized_attention_weights_A_q, bilstm_encoded_contexts), perm = [0, 2, 1])

        # Dimensionalities:
        # Q_C_q_concat: samples x 2*2*n_hidden_enc * question_words
        # coattention_context_C_d: samples x 2*2*n_hidden_enc * context_words
        bilstm_encoded_questions_transpose = tf.transpose(bilstm_encoded_questions, perm = [0, 2, 1])
        Q_C_q_concat = tf.concat([bilstm_encoded_questions_transpose, attention_contexts_C_q], 1)
        A_d_transpose = tf.transpose(normalized_attention_weights_A_d, perm = [0, 2, 1])
        coattention_context_C_d = tf.matmul(Q_C_q_concat, A_d_transpose)

        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_mix, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_mix, forget_bias=1.0)
        
        # Dimensionalities:
        # D_C_d: samples x context_words x 3*2*n_hidden_enc
        # U: samples x context_words x 2*n_hidden_mix
        C_d_transpose = tf.transpose(coattention_context_C_d, perm = [0, 2, 1])
        D_C_d = tf.concat([bilstm_encoded_contexts, C_d_transpose], 2)
        U, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, D_C_d, sequence_length=context_lengths, dtype=tf.float64)
        return tf.concat(U, 2) #U 

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.n_hidden_dec = 50

    def decode(self, coattention_encoding, context_lengths):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        # Dimensionalities:
        # coattention_encoding: samples x context_words x 2*n_hidden_mix (it's packed in like this: ((data)), for some reason)
        # return value: samples x context_words x 2*n_hidden_dec

        with tf.variable_scope("DecoderBiLSTM"):
            # Forward direction cell
            decoder_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_dec, forget_bias=1.0)
            # Backward direction cell
            decoder_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_dec, forget_bias=1.0)

            decoder_output, _, = tf.nn.bidirectional_dynamic_rnn(decoder_lstm_fw_cell, decoder_lstm_bw_cell, coattention_encoding,
                                                  sequence_length=context_lengths, dtype=tf.float64)

        return tf.concat(decoder_output, 2)

class QASystem(object):
    def __init__(self, encoder, decoder, mixer, embed_path):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.mixer = mixer
        self.decoder = decoder
        self.pretrained_embeddings = np.load(embed_path)["glove"]

        # ==== set up placeholder tokens ========

        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.questions_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, 2))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            question_embeddings_lookup, context_embeddings_lookup = self.setup_embeddings()
            self.setup_system(
                question_embeddings_lookup,
                context_embeddings_lookup
            )
            #self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self, question_embeddings_lookup, context_embeddings_lookup):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        self.question_embeddings_lookup = question_embeddings_lookup
        self.context_embeddings_lookup = context_embeddings_lookup

        bilstm_encoded_questions, bilstm_encoded_contexts = self.encoder.encode(
            question_embeddings_lookup,
            self.questions_lengths_placeholder,
            context_embeddings_lookup,
            self.context_lengths_placeholder
        )

        self.bilstm_encoded_questions = bilstm_encoded_questions
        self.bilstm_encoded_contexts = bilstm_encoded_contexts

        self.coattention_encoding = self.mixer.mix(bilstm_encoded_questions, bilstm_encoded_contexts, self.context_lengths_placeholder)
        self.network = self.decoder.decode(self.coattention_encoding, self.context_lengths_placeholder)

#
#        raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings)
            question_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            context_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            return question_embeddings_lookup, context_embeddings_lookup

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def test_encoders_and_mixer(self, session, dataset):
        feed_dict = {
            self.question_placeholder: dataset['questions'],
            self.questions_lengths_placeholder: dataset['question_lengths'],
            self.context_placeholder: dataset['contexts'],
            self.context_lengths_placeholder: dataset['context_lengths'],
        }

        # Dimensionalities:
        # question_placeholder: samples x words
        # context_placeholder: samples x words
        # question_embeddings_lookup: samples x words x embed_size
        # context_embeddings_lookup: samples x words x embed_size
        # bilstm_encoded_questions: samples x words x 2*n_hidden_enc
        # bilstm_encoded_contexts: samples x words x 2*n_hidden_enc

        # print_debug_output = tf.Print(self.network, [self.network], summarize=500)
        
        out1 = session.run([self.network], feed_dict)
        print("Final layer shape:", out1[0].shape)
        #out1, out2 = session.run([self.bilstm_encoded_questions, self.bilstm_encoded_contexts], feed_dict)        
        
        #print("dataset['questions']:", dataset['questions'])
        #print("question_placeholder", self.question_placeholder.get_shape())
        #print("bilmst_enc_qs", self.bilstm_encoded_questions.get_shape())

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
