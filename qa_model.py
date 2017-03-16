from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random
import os

import numpy as np
import tensorflow as tf

from data_utils import open_dataset, split_in_batches
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops import variable_scope as vs
from pdb import set_trace as t
from evaluate import exact_match_score, f1_score
from contrib_ops import highway_maxout, batch_linear

logging.basicConfig(level=logging.INFO)

def lengths_to_masks(lengths, max_length):
    """
    arguments: lengths: a lengths placeholder of (batch_size,)
               max_length: maximum token lengths
    returns:   masks: A masking matrix of size (batch_size, max_length)
    """
    lengths = tf.reshape(lengths, [-1])
    tiled_ranges = tf.tile(
        tf.expand_dims(tf.range(max_length), 0), [tf.shape(lengths)[0], 1])
    lengths = tf.expand_dims(lengths, 1)
    masks = tf.to_float(tf.to_int64(tiled_ranges) < tf.to_int64(lengths))
    return masks

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

# jorisvanmens: some of these might get overwritten in the relevant functions (would be good to fix)
class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """


    def __init__(self, FLAGS):
        self.test = FLAGS.test
        self.evaluate = FLAGS.evaluate
        self.learning_rate = FLAGS.learning_rate
        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.dropout = FLAGS.dropout
        self.batch_size = FLAGS.batch_size
        self.epochs = FLAGS.epochs
        self.state_size = FLAGS.state_size
        self.output_size = FLAGS.output_size
        self.embedding_size = FLAGS.embedding_size
        self.n_hidden_enc = FLAGS.n_hidden_enc
        self.n_hidden_mix = FLAGS.n_hidden_mix
        self.n_hidden_dec_base = FLAGS.n_hidden_dec_base
        self.n_hidden_dec_hmn = FLAGS.n_hidden_dec_hmn
        self.max_examples = FLAGS.max_examples
        self.maxout_size = FLAGS.maxout_size
        self.max_decode_steps = FLAGS.max_decode_steps
        self.batches_per_save = FLAGS.batches_per_save
        self.after_each_batch = FLAGS.after_each_batch
        self.data_dir = FLAGS.data_dir
        self.train_dir = FLAGS.train_dir
        self.load_train_dir = FLAGS.load_train_dir
        self.log_dir = FLAGS.log_dir
        self.optimizer = FLAGS.optimizer
        self.print_every = FLAGS.print_every
        self.keep = FLAGS.keep
        self.vocab_path = FLAGS.vocab_path
        self.embed_path = FLAGS.embed_path
        self.model = FLAGS.model

def _createBiLSTM(embeddings, sequence_length, hidden_size, dropout, initial_state=None):
    if initial_state is not None:
        initial_state_fw = initial_state[0]
        initial_state_bw = initial_state[1]
    else:
        initial_state_fw = None
        initial_state_bw = None

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    embeddings_drop = tf.nn.dropout(embeddings, dropout)
    hidden_state, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                              lstm_bw_cell,
                                              embeddings_drop,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              sequence_length=sequence_length,
                                              dtype=tf.float32)

    return tf.concat(hidden_state, 2), final_state

class BiLSTMEncoder(object):
    # jorisvanmens: encodes question and context using a BiLSTM (code by Ilya)
    def __init__(self, config):
        self.config = config


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
        with tf.variable_scope("Encoder"):
            return _createBiLSTM(embeddings, sequence_length, self.config.n_hidden_enc, self.config.dropout, initial_state)

class LSTMEncoder(object):
    def __init__(self, config):
        self.config = config

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
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_enc, forget_bias=1.0)
        hidden_state, final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                  embeddings,
                                                  initial_state=initial_state,
                                                  sequence_length=sequence_length,
                                                  dtype=tf.float32)

        return hidden_state, final_state

class FFNN(object):
    def __init__(self, input_size, output_size, hidden_size):
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

        weights1 = tf.get_variable('weights1', shape=(self.input_size, self.hidden_size), initializer=initializer, dtype=tf.float32)
        biases1 = tf.Variable(tf.zeros((1, self.hidden_size), tf.float32))
        weights2 = tf.get_variable('weights2', shape=(self.hidden_size, self.output_size), initializer=initializer, dtype=tf.float32)
        if self.output_size > 1: # don't need bias if output_size == 1
            biases2 = tf.Variable(tf.zeros((1, self.output_size), tf.float32))

        h = tf.nn.relu(tf.matmul(inputs, weights1) + biases1) # samples x n_hidden_dec
        h_drop = tf.nn.dropout(h, dropout_placeholder)
        if self.output_size > 1:
            output = tf.matmul(h_drop, weights2) + biases2
        else:
            output = tf.matmul(h_drop, weights2) # don't need bias if output_size == 1
        return output # samples x context_words

class Mixer(object):
    # jorisvanmens: creates coattention matrix from encoded question and context (code by Joris)

    def __init__(self, config):
            self.config = config

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

        # Dimensionalities:
        # D_C_d: samples x context_words x 3*2*n_hidden_enc
        # U: samples x context_words x 2*n_hidden_mix
        C_d_transpose = tf.transpose(coattention_context_C_d, perm = [0, 2, 1])
        D_C_d = tf.concat([bilstm_encoded_contexts, C_d_transpose], 2)

        with tf.variable_scope("Mixer"):
            U, U_final = _createBiLSTM(D_C_d, context_lengths, self.config.n_hidden_mix, self.config.dropout)

        # This gets the final forward & backward hidden states from the output, and concatenates them
        U_final_hidden = tf.concat((U_final[0].h, U_final[1].h), 1)
        return U, U_final_hidden

class Decoder(object):
    # jorisvanmens: decodes coattention matrix using a simple neural net (code by Joris)

    def __init__(self, config):
        self.config = config

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

        if self.config.model == "baseline-v4":
            USE_DECODER_VERSION = 4
        elif self.config.model == "baseline-v3":
            USE_DECODER_VERSION = 3
        else:
            USE_DECODER_VERSION = 2
        logging.info("Using decoder version %d" % USE_DECODER_VERSION)


        if USE_DECODER_VERSION == 4:
            # Similar to BiDAF paper
            # For start_pred, just use a vector (like V2)
            # For end_pred, adding an additional BiLSTM

            weights_shape = (n_hidden_mix, 1)
            #bnew_shape = (1)

            Ureshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])

            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("StartPredictor"):
                decoder_bilstm_output, _ = _createBiLSTM(coattention_encoding, context_lengths, self.config.n_hidden_dec_base, self.config.dropout)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])
                start_pred_tmp = tf.matmul(decoder_bilstm_output_reshape, weights)# + bnew
                start_pred_tmp2 = tf.reshape(start_pred_tmp, [-1, self.config.output_size])
                start_pred = start_pred_tmp2# + bnew

            with tf.variable_scope("EndPredictor"):
                decoder_bilstm_output, _ = _createBiLSTM(coattention_encoding, context_lengths, self.config.n_hidden_dec_base, self.config.dropout)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])
                end_pred_tmp = tf.matmul(decoder_bilstm_output_reshape, weights)# + bnew
                end_pred_tmp2 = tf.reshape(end_pred_tmp, [-1, self.config.output_size])
                end_pred = end_pred_tmp2# + bnew

        elif USE_DECODER_VERSION == 3:
            # This decoder also uses the full coattention matrix as input
            # It then takes a matrix coattention column (corresponding to a single context word)
            # And throws it into a simple FFNN
            Ureshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])
            output_size = 1

            with tf.variable_scope("StartPredictor"):
                start_ffnn = FFNN(n_hidden_mix, output_size, self.config.n_hidden_dec_base)
                start_pred_tmp = start_ffnn.forward_prop(Ureshape, dropout_placeholder)
                start_pred = tf.reshape(start_pred_tmp, [-1, self.config.output_size])

            with tf.variable_scope("EndPredictor"):
                end_ffnn = FFNN(n_hidden_mix, output_size, self.config.n_hidden_dec_base)
                end_pred_tmp = end_ffnn.forward_prop(Ureshape, dropout_placeholder)
                end_pred = tf.reshape(start_pred_tmp, [-1, self.config.output_size])


        elif USE_DECODER_VERSION == 2:
            # This decoder uses the full coattention matrix as input
            # Multiplies a single vector to every coattention matrix's column (corresponding to a single context word)
            # and adds biases to create logits

            weights_shape = (n_hidden_mix, 1)
            #bnew_shape = (1)

            Ureshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])

            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("StartPredictor"):
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                start_pred_tmp = tf.matmul(Ureshape, weights)# + bnew
                start_pred_tmp2 = tf.reshape(start_pred_tmp, [-1, self.config.output_size])
                start_pred = start_pred_tmp2# + bnew

            with tf.variable_scope("EndPredictor"):
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                end_pred_tmp = tf.matmul(Ureshape, weights)# + bnew
                end_pred_tmp2 = tf.reshape(end_pred_tmp, [-1, self.config.output_size])
                end_pred = end_pred_tmp2# + bnew

        else:
            # This uses only the final hidden layer from the coattention matrix,
            # and feeds it into a simple neural net
            ffnn = FFNN(self.config.n_hidden_mix * 2, self.config.n_hidden_dec_base, self.config.output_size)

            with tf.variable_scope("StartPredictor"):
                start_pred = ffnn.forward_prop(coattention_encoding_final_states, dropout_placeholder)

            with tf.variable_scope("EndPredictor"):
                end_pred = ffnn.forward_prop(coattention_encoding_final_states, dropout_placeholder)

        return start_pred, end_pred

class HMNDecoder(object):
    # jorisvanmens: decodes coattention matrix using a complex Highway model (code by Ilya)
    # based on co-attention paper
    def __init__(self, config):
        self.config = config

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
        maxout_size = self.config.maxout_size
        max_decode_steps = self.config.max_decode_steps
        self._initial_guess = np.zeros((2, self.config.batch_size), dtype=np.int32)
        self._u = coattention_encoding

        def select(u, pos, idx):
              # u: (samples x context_words x 2 * n_hidden_mix)
              # sample: (context_words x 2 * n_hidden_mix)
              sample = tf.gather(u, idx)

              # u_t: (2 * n_hidden_mix)
              pos_idx = tf.gather(tf.reshape(pos, [-1]), idx)

              u_t = tf.gather( sample, pos_idx)
              return u_t

        with tf.variable_scope('selector'):
            # LSTM for decoding

            lstm_dec = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_hmn)
            # init highway fn
            highway_alpha = highway_maxout(self.config.n_hidden_dec_hmn, maxout_size)
            highway_beta = highway_maxout(self.config.n_hidden_dec_hmn, maxout_size)

            # _u dimension: (batch_size, context, 2*self.config.n_hidden_dec_hmn)
            # reshape self._u to (context, batch_size, 2*self.config.n_hidden_dec_hmn)
            U = tf.transpose(self._u[:,:self.config.output_size,:], perm=[1, 0, 2])

            # batch indices
            loop_until = tf.to_int32(np.array(range(self.config.batch_size)))
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
                  s = tf.reshape(tf.argmax(alpha, 0), [self.config.batch_size])
                  # update start guess
                  fn = lambda idx: select(self._u, s, idx)
                  u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                with tf.variable_scope('highway_beta'):
                  # compute end position next
                  fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                  beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                  e = tf.reshape(tf.argmax(beta, 0), [self.config.batch_size])
                  # update end guess
                  fn = lambda idx: select(self._u, e, idx)
                  u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                self._s.append(s)
                self._e.append(e)
                self._alpha.append(tf.reshape(alpha, [self.config.batch_size, -1]))
                self._beta.append(tf.reshape(beta, [self.config.batch_size, -1]))
        return self._alpha, self._beta

class QASystem(object):
    def __init__(self, encoder, decoder, mixer, embed_path, config, model="baseline"):
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
        self.model = model

        # ==== set up placeholder tokens ========

        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.questions_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.output_size))
        self.context_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.answers_numeric_list = tf.placeholder(tf.int32, shape=(None, 2))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            if model == 'baseline' or model == 'baseline-v2' or model == 'baseline-v3' or model == 'baseline-v4':
                self.setup_loss()
            else:
                self.setup_hmn_loss()
            self.setup_train_op()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()


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
        """
        Set up your loss computation here
        :return:
        """

        mask = lengths_to_masks(self.context_lengths_placeholder, self.config.output_size)

        masked_start_preds = mask * self.start_prediction
        masked_end_preds = mask * self.end_prediction

        sparse_start_labels = self.answers_numeric_list[:, 0]
        sparse_end_labels = self.answers_numeric_list[:, 1]

        start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_start_preds, labels=sparse_start_labels)
        end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_end_preds, labels=sparse_end_labels)

        L2_factor = 0.001
        L2_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if 'weight' in tensor.name ]) * L2_factor

        self.loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss) + L2_loss

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
            with tf.variable_scope("loss"):
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

        output_feed = [self.loss]

        out = session.run(output_feed, input_feed)

        return out[0]

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

    def validate(self, sess, valid_dataset_batches):
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

        for valid_x, valid_y in valid_dataset_batches:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=True):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        TODO(nwestman): Create function to map id's back to words.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        random.shuffle(dataset)

        # cap number of samples
        dataset = dataset[:sample]

        questions, question_lengths, contexts, context_lengths, answers = open_dataset(dataset)
        data_batches = split_in_batches(questions, question_lengths, contexts, context_lengths, self.config.batch_size, answers=answers)

        for batch_idx, (test_batch_x, test_batch_y) in enumerate(data_batches):
            logging.info("Evaluating batch %s of %s" % (batch_idx, len(data_batches)))
            valid_loss = self.test(session, test_batch_x, test_batch_y)
            answers_numeric_list = test_batch_y
            answer_start_predictions, answer_end_predictions = self.answer(session, test_batch_x)

            for idx, answer_indices in enumerate(answers_numeric_list):
                context = test_batch_x['contexts'][idx]

                answer_indices = map(int, answer_indices)
                prediction_indices = [answer_start_predictions[idx], answer_end_predictions[idx]]

                ground_truth_ids = ' '.join(context[answer_indices[0]: answer_indices[1] + 1])
                prediction_ids = ' '.join(context[prediction_indices[0]: prediction_indices[1] + 1])

                f1 += f1_score(prediction_ids, ground_truth_ids)
                em += exact_match_score(prediction_ids, ground_truth_ids)

        em = 100.0 * em / sample
        f1 = 100.0 * f1 / sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em, valid_loss

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

    def train(self, session, dataset, train_dir):
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

        save_path = os.path.join(train_dir, self.model)

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            var_params = variable.get_shape().num_elements()
            total_parameters = total_parameters + var_params
            logging.info("Tensor %s has shape %s with %d parameters" % (variable.name, str(shape), var_params))
        logging.info("%d total parameters" % total_parameters)

        for epoch in xrange(self.config.epochs):
            logging.info("Starting epoch %d", epoch)
            random.shuffle(dataset['train']) # Make sure to shuffle the dataset.
            questions, question_lengths, contexts, context_lengths, answers = open_dataset(dataset['train'])
            data_batches = split_in_batches(questions, question_lengths, contexts, context_lengths, self.config.batch_size, answers=answers)
            for idx, (batch_x, batch_y) in enumerate(data_batches):
                tic = time.time()
                loss = self.optimize(session, batch_x, batch_y)
                toc = time.time()
                logging.info("Batch %s processed in %s seconds." % (str(idx), format(toc - tic, '.2f')))
                logging.info("Training loss: %s" % format(loss, '.5f'))
                if (idx + 1) % self.config.batches_per_save == 0 or self.config.test:
                    logging.info("Saving model after batch %s" % str(idx))
                    tic = time.time()
                    checkpoint_path = self.saver.save(session, save_path)
                    tf.train.update_checkpoint_state(train_dir, checkpoint_path)
                    toc = time.time()
                    logging.info("Saved in %s seconds" % format(toc - tic, '.2f'))

                if (idx + 1) % self.config.after_each_batch == 0 or self.config.test:
                    _, _, valid_loss = self.evaluate_answer(session, dataset['val'])
                    logging.info("Sample validation loss: %s" % format(valid_loss, '.5f'))
                    if self.config.test: #test the graph
                        logging.info("Graph successfully executes.")

            logging.info("Evaluating current model..")
            _, _, valid_loss = self.evaluate_answer(session, dataset['val'], len(dataset['val']))
            logging.info("Validation loss: %s" % format(valid_loss, '.5f'))
            _, _, valid_loss = self.evaluate_answer(session, dataset['train'], len(dataset['val'])) #subset of full dataset for speed
            logging.info("Train loss: %s" % format(valid_loss, '.5f'))
