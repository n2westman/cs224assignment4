from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random

import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops import variable_scope as vs
from pdb import set_trace as t
from evaluate import exact_match_score, f1_score
from contrib_ops import highway_maxout, batch_linear
from random import shuffle

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

hidden_size = 200
maxout_size = 32
max_timesteps = 600
max_decode_steps = 4

# jorisvanmens: some of these might get overwritten in the relevant functions (would be good to fix)
class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    def __init__(self, batch_size=100, optimizer="adam", learning_rate=0.001, dropout=0.15):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout

class Encoder(object):
    # jorisvanmens: encodes question and context using a BiLSTM (code by Ilya)
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim
        self.n_hidden_enc = 200

    def encode(self, embeddings, sequence_length, initial_state=None):
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

        if initial_state is not None:
            initial_state_fw = initial_state[0]
            initial_state_bw = initial_state[1]
        else:
            initial_state_fw = None
            initial_state_bw = None

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
        hidden_state, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                      lstm_bw_cell,
                                                      embeddings,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sequence_length,
                                                      dtype=tf.float32)
        return tf.concat(hidden_state, 2), final_state

class FFNN(object):
    def __init__(self, input_size, output_size, hidden_size=200):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward_prop(self, inputs, dropout_placeholder):
        """
        General 1-layer FFNN.

        TODO(nwestman): Turn dropout off at test time.

        :return: predictions
        """
        initializer = tf.contrib.layers.xavier_initializer()

        W1 = tf.get_variable('W1', shape=(self.input_size, self.hidden_size), initializer=initializer, dtype=tf.float32)
        b1 = tf.Variable(tf.zeros((1, self.hidden_size), tf.float32))
        W2 = tf.get_variable('W2', shape=(self.hidden_size, self.output_size), initializer=initializer, dtype=tf.float32)
        b2 = tf.Variable(tf.zeros((1, self.output_size), tf.float32))

        h = tf.nn.relu(tf.matmul(inputs, W1) + b1) # samples x n_hidden_dec
        h_drop = tf.nn.dropout(h, dropout_placeholder)
        return tf.matmul(h_drop, W2) + b2 # samples x context_words

class Mixer(object):
    # jorisvanmens: creates coattention matrix from encoded question and context (code by Joris)

    def __init__(self):
            self.n_hidden_mix = 200

    def mix(self, bilstm_encoded_questions, bilstm_encoded_contexts, context_lengths):
        # Compute the attention on each word in the context as a dot product of its contextual embedding and the query

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

        U, U_final = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, D_C_d, sequence_length=context_lengths, dtype=tf.float32)
        # This gets the final forward & backward hidden states from the output, and concatenates them
        U_final_hidden = tf.concat((U_final[0].h, U_final[1].h), 1)
        return tf.concat(U, 2), U_final_hidden

class Decoder(object):
    # jorisvanmens: decodes coattention matrix using a simple neural net (code by Joris)

    def __init__(self, output_size, batch_size):
        self.output_size = output_size
        self.n_hidden_dec = 20
        self.batch_size = batch_size

    def decode(self, coattention_encoding, coattention_encoding_final_states, context_lengths, dropout_placeholder):
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
        # coattention_encoding: samples x context_words x 2*n_hidden_mix
        # coattention_encoding_final_states: samples x 2*n_hidden_mix
        # decoder_output_concat: samples x context_words x 2*n_hidden_dec

        num_samples = coattention_encoding.get_shape()[0]
        max_context_words = coattention_encoding.get_shape()[1]
        n_hidden_mix = coattention_encoding.get_shape()[2]

        # What do we want to do here? Create a simple regression / single layer neural net
        # We have U_final = samples x 2*n_hidden_mix input
        # We want to do h = relu(U * W + b1)
        # Here, W has to be 2*n_hidden_mix x n_hidden_dec
        # b has to be n_hidden_dec

        U = coattention_encoding
        U_final = coattention_encoding_final_states

        USE_DECODER_VERSION = 2
        print("Using decoder version", USE_DECODER_VERSION)

        if USE_DECODER_VERSION == 3:
            # This decoder also uses the full coattention matrix as input
            # It then takes a matrix coattention column (corresponding to a single context word)
            # And throws it into a simple FFNN
            Ureshape = tf.reshape(U, [-1, 2 * hidden_size])
            output_size = 1
            ffnn = FFNN(n_hidden_mix, output_size, self.n_hidden_dec)

            with tf.variable_scope("StartPredictor"):                
                start_pred_tmp = ffnn.forward_prop(Ureshape, dropout_placeholder)
                start_pred = tf.reshape(start_pred_tmp, [-1, max_timesteps])

            with tf.variable_scope("EndPredictor"):
                end_pred_tmp = ffnn.forward_prop(Ureshape, dropout_placeholder)
                end_pred = tf.reshape(start_pred_tmp, [-1, max_timesteps])


        elif USE_DECODER_VERSION == 2:
            # This decoder uses the full coattention matrix as input
            # Multiplies a single vector to every coattention matrix's column (corresponding to a single context word)
            # and adds biases to create logits

            Wnew_shape = (n_hidden_mix, 1)
            #bnew_shape = (1)
            Ureshape = tf.reshape(U, [-1, 2 * hidden_size])
            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("StartPredictor"):
                Wnew = tf.get_variable('Wnew', shape=Wnew_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                start_pred_tmp = tf.matmul(Ureshape, Wnew)# + bnew
                start_pred_tmp2 = tf.reshape(start_pred_tmp, [-1, max_timesteps])
                start_pred = start_pred_tmp2# + bnew

            with tf.variable_scope("EndPredictor"):
                Wnew = tf.get_variable('Wnew', shape=Wnew_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                end_pred_tmp = tf.matmul(Ureshape, Wnew)# + bnew
                end_pred_tmp2 = tf.reshape(end_pred_tmp, [-1, max_timesteps])
                end_pred = end_pred_tmp2# + bnew

        else:
            # This uses only the final hidden layer from the coattention matrix,
            # and feeds it into a simple neural net

            num_samples = coattention_encoding.get_shape()[0]
            max_context_words = coattention_encoding.get_shape()[1]
            n_hidden_mix = coattention_encoding.get_shape()[2]

            ffnn = FFNN(n_hidden_mix, self.n_hidden_dec, max_context_words)

            with tf.variable_scope("StartPredictor"):
                start_pred = ffnn.forward_prop(coattention_encoding_final_states, dropout_placeholder)

            with tf.variable_scope("EndPredictor"):
                end_pred = ffnn.forward_prop(coattention_encoding_final_states, dropout_placeholder)

        return start_pred, end_pred


class HMNDecoder(object):
    # jorisvanmens: decodes coattention matrix using a complex Highway model (code by Ilya)
    # based on co-attention paper

    def __init__(self, output_size, batch_size):
        self.output_size = output_size
        self.n_hidden_dec = 50
        self.batch_size = batch_size

    def decode(self, coattention_encoding, coattention_encoding_final_states, context_lengths, dropout):
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
        # coattention_encoding: samples x context_words x 2*n_hidden_mix
        # return value: samples x context_words x 2*n_hidden_dec
        self._initial_guess = np.zeros((2, self.batch_size), dtype=np.int32)
        self._u = coattention_encoding
        print("_u", self._u.get_shape())

        def select(u, pos, idx):
              # u: (samples x context_words x 2 * n_hidden_mix)
              # sample: (context_words x 2 * n_hidden_mix)
              sample = tf.gather(u, idx)
              print("sample", sample)
              # u_t: (2 * n_hidden_mix)
              pos_idx = tf.gather(tf.reshape(pos, [-1]), idx)
              print("pos", pos)
              print("pos_idx", pos_idx)
              u_t = tf.gather( sample, pos_idx)

              print("u_t", u_t.get_shape())
              #reshaped_u_t = tf.reshape( u_t, [-1])
              return u_t

        with tf.variable_scope('selector'):
            # LSTM for decoding
            lstm_dec = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            # init highway fn
            highway_alpha = highway_maxout(hidden_size, maxout_size)
            highway_beta = highway_maxout(hidden_size, maxout_size)

            # _u dimension: (batch_size, context, 2*hidden_size)
            # reshape self._u to (context, batch_size, 2*hidden_size)
            U = tf.transpose(self._u[:,:max_timesteps,:], perm=[1, 0, 2])

            # batch indices
            loop_until = tf.to_int32(np.array(range(self.batch_size)))
            # initial estimated positions
            # s and e have dimension [self.batch_size]
            s, e = tf.split(self._initial_guess, 2, 0)

            fn = lambda idx: select(self._u, s, idx)
            u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
            print( "u_s", u_s)

            fn = lambda idx: select(self._u, e, idx)
            u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
            print( "u_e", u_e)

        self._s, self._e = [], []
        self._alpha, self._beta = [], []
        with tf.variable_scope("Decoder") as scope:
            for step in range(max_decode_steps):
                if step > 0: scope.reuse_variables()
                # single step lstm
                _input = tf.concat([u_s, u_e], 1)

                print( "_input:", _input)
                # Note: This is a single-step rnn.
                # static_rnn does not need a time step dimension in input.
                _, h = tf.contrib.rnn.static_rnn(lstm_dec, [_input], dtype=tf.float32)
                # Note: h is the output state of the last layer which
                # includes a tuple: (output, hidden state), which is concatenated along second axis.
                print("h", h)
                #print("st", st)
                h_state = tf.concat(h, 1)
                print("h_state", h_state)

                with tf.variable_scope('highway_alpha'):
                  # compute start position first
                  print("u_s", u_s)
                  fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e)
                  alpha = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                  s = tf.reshape(tf.argmax(alpha, 0), [self.batch_size])
                  # update start guess
                  fn = lambda idx: select(self._u, s, idx)
                  u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                with tf.variable_scope('highway_beta'):
                  # compute end position next
                  fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                  beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                  e = tf.reshape(tf.argmax(beta, 0), [self.batch_size])
                  # update end guess
                  fn = lambda idx: select(self._u, e, idx)
                  u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                self._s.append(s)
                self._e.append(e)
                self._alpha.append(tf.reshape(alpha, [self.batch_size, -1]))
                self._beta.append(tf.reshape(beta, [self.batch_size, -1]))
        return self._alpha, self._beta

class QASystem(object):
    def __init__(self, encoder, decoder, mixer, embed_path, max_context_length, config, model):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.mixer = mixer
        self.decoder = decoder
        self.config = config
        self.pretrained_embeddings = np.load(embed_path)["glove"]

        # ==== set up placeholder tokens ========

        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.questions_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, max_context_length))
        self.context_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.answers_numeric_list = tf.placeholder(tf.int32, shape=(None, 2))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            if model == 'baseline':
                self.setup_loss()
            else:
                self.setup_hmn_loss()
            self.setup_train_op()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()

    def split_in_batches(self, dataset):
        # jorisvanmens: splits a dataset into batches of batch_size (code by Joris)

        batches = []
        for start_index in range(0, len(dataset['questions']), self.config.batch_size):
            batch_x = {
                'questions': dataset['questions'][start_index:start_index + self.config.batch_size],
                'question_lengths': dataset['question_lengths'][start_index:start_index + self.config.batch_size],
                'contexts': dataset['contexts'][start_index:start_index + self.config.batch_size],
                'context_lengths': dataset['context_lengths'][start_index:start_index + self.config.batch_size]
            }
            batch_y = dataset['answers_numeric_list'][start_index:start_index + self.config.batch_size]
            batches.append((batch_x, batch_y))

        print("Created", str(len(batches)), "batches")
        return batches


    def setup_system(self):
        # jorisvanmens: sets up parts of the graph (code by Ilya & Joris)
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        with tf.variable_scope("q"):
            bilstm_encoded_questions, encoded_question_final_state = self.encoder.encode(self.question_embeddings_lookup, self.questions_lengths_placeholder)

        with tf.variable_scope("c"):
            bilstm_encoded_contexts, _ = self.encoder.encode(self.context_embeddings_lookup, self.context_lengths_placeholder, encoded_question_final_state)

        coattention_encoding, coattention_encoding_final_states \
            = self.mixer.mix(bilstm_encoded_questions, bilstm_encoded_contexts, self.context_lengths_placeholder)
        self.start_prediction, self.end_prediction = \
            self.decoder.decode(coattention_encoding, coattention_encoding_final_states, self.context_lengths_placeholder, self.dropout_placeholder)


    def setup_loss(self):
        # jorisvanmens: calculates loss for the neural net decoder (code by Joris)
        # jorisvanmens: this is not tested at all (like most parts of the code really, haha)
        """
        Set up your loss computation here
        :return:
        """
        sm_ce_loss_answer_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.start_prediction, labels = self.answers_numeric_list[:, 0])
        sm_ce_loss_answer_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.end_prediction, labels = self.answers_numeric_list[:, 1])
        self.loss = tf.reduce_mean(sm_ce_loss_answer_start) + tf.reduce_mean(sm_ce_loss_answer_end)

    def setup_hmn_loss(self):
        # jorisvanmens: calculates loss for the HMN decoder (code by Ilya)
        # based on co-attention paper
        def _loss_shared(logits, labels):
          labels = tf.Print( labels, [tf.shape(labels)] )
          labels = tf.reshape(labels, [self.config.batch_size])
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=labels, name='per_step_cross_entropy')
          cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
          tf.add_to_collection('per_step_losses', cross_entropy_mean)
          return tf.add_n(tf.get_collection('per_step_losses'), name='per_step_loss')

        def _loss_multitask(logits_alpha, labels_alpha,
                          logits_beta, labels_beta):
            """Cumulative loss for start and end positions."""
            with vs.variable_scope("loss"):
                fn = lambda logit, label: _loss_shared(logit, label)
                loss_alpha = [fn(alpha, labels_alpha) for alpha in logits_alpha]
                loss_beta = [fn(beta, labels_beta) for beta in logits_beta]
                return tf.reduce_sum([loss_alpha, loss_beta], name='loss')

        alpha_true, beta_true = tf.split(self.answers_numeric_list, 2, 0)
        self.loss = _loss_multitask(self.decoder._alpha, alpha_true,
                                    self.decoder._beta, beta_true)


    def setup_embeddings(self):
        # jorisvanmens: looks up embeddings (code by Ilya)
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.constant(self.pretrained_embeddings, dtype=tf.float32)
            self.question_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            self.context_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.context_placeholder)

    def setup_train_op(self):
        optimizer = get_optimizer(self.config.optimizer)
        self.train_op = optimizer(self.config.learning_rate).minimize(self.loss)

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return: loss - the training loss
        """
        input_feed = self.create_feed_dict(train_x, train_y, 1.0 - self.config.dropout)

        output_feed = [self.train_op, self.loss]

        _, loss = session.run(output_feed, input_feed)

        return loss

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return: loss - the validation loss
        """
        input_feed = self.create_feed_dict(valid_x, valid_y)

        output_feed = [self.train_op, self.loss]

        _, loss = session.run(output_feed, input_feed)

        return loss

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.create_feed_dict(test_x)

        output_feed = [self.start_prediction, self.end_prediction]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):
        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        # jorisvanmens: prefab code, not used
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

    def evaluate_answer(self, session, data_batches, sample=100, log=True):
        # jorisvanmens: calculate F1 and EM on a random batch (code by Joris)
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

        test_batch_x, test_batch_y = random.choice(data_batches)
        answers_numeric_list = test_batch_y
        answer_start_predictions, answer_end_predictions = self.answer(session, test_batch_x)

        f1s = []
        ems = []

        for idx, answer_numeric in enumerate(answers_numeric_list):
            answer_numeric = map(int, answer_numeric)
            prediction = [answer_start_predictions[idx], answer_end_predictions[idx]]

            em = 0.
            if prediction[0] == answer_numeric[0] and prediction[1] == answer_numeric[1]:
                em = 1.
            f1 = 0.
            prediction_range = range(prediction[0], prediction[1] + 1)
            answer_range = range(answer_numeric[0], answer_numeric[1] + 1)
            num_same = len(set(prediction_range) & set(answer_range))
            if len(prediction_range) == 0:
                precision = 0
            else:
                precision = 1.0 * num_same / len(prediction_range)
            if len(answer_range) == 0:
                recall = 0
            else:
                recall = 1.0 * num_same / len(answer_range)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            f1s.append(f1)
            ems.append(em)

        f1 = sum(f1s) / len(f1s) * 100
        em = sum(ems) / len(ems) * 100

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def create_feed_dict(self, batch_x, batch_y=None, dropout=1.0):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            batch: A batch of input / output data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.question_placeholder: batch_x['questions'],
            self.questions_lengths_placeholder: batch_x['question_lengths'],
            self.context_placeholder: batch_x['contexts'],
            self.context_lengths_placeholder: batch_x['context_lengths'],
            self.dropout_placeholder: dropout
        }
        if batch_y is not None:
            feed_dict[self.answers_numeric_list] = batch_y

        return feed_dict

    def train(self, session, dataset, train_dir, test=False):
        # jorisvanmens: actual training function, this is where the time is spent (code by Joris)
        # needs a lot more work
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
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

        num_epochs = 10
        # TODO(nwestman): move to a flag
        after_each_batch = 10 # Note one evaluation takes as much time
        data_batches = self.split_in_batches(dataset['train'])
        test_data_batches = self.split_in_batches(dataset['val'])

        # Block of prefab code that check the number of params
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for epoch in xrange(num_epochs):
            shuffle(data_batches)
            for idx, (batch_x, batch_y) in enumerate(data_batches):
                tic = time.time()
                loss = self.optimize(session, batch_x, batch_y)
                toc = time.time()
                logging.info("Batch %s processed in %s seconds." % (str(idx), format(toc - tic, '.2f')))
                logging.info("Training loss: %s" % format(loss, '.5f'))
                if (idx + 1) % after_each_batch == 0:
                    f1, em = self.evaluate_answer(session, test_data_batches)
                    if test: #test the graph
                        logging.info("Graph successfully executes.")
                        break
            checkpoint_path = self.saver.save(session, train_dir)
            tf.train.update_checkpoint_state(train_dir, checkpoint_path)
