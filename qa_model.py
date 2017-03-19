from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import google3
    GOOGLE3 = True
except ImportError:
    GOOGLE3 = False

if GOOGLE3:
    from google3.pyglib import gfile
    from google3.experimental.users.ikuleshov.cs224n.evaluate import exact_match_score, f1_score
    from google3.experimental.users.ikuleshov.cs224n.contrib_ops import highway_maxout
    from google3.experimental.users.ikuleshov.cs224n.data_utils import open_dataset, split_in_batches
else:
    from evaluate import exact_match_score, f1_score
    from contrib_ops import highway_maxout
    from data_utils import open_dataset, split_in_batches, make_prediction_plot

import time
import logging
import random
import os

import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from pdb import set_trace as t

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

def masked_loss(logits, labels, mask):
    masked_logits = tf.multiply(mask, logits)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_logits, labels=labels))

def batch_slice(params, indices):
    """
    Grabs a list of slices along the second axis in a tensor.

    Similar to tf.gather(), but along axis=1.

    arguments: params: A `Tensor` of params (batch_size, dim1, dim2, dim3 ...)
               indices: Indices
    returns:   masks: A `Tensor` of params (batch_size, dim2, dim3, ...)
    """
    dim_size = tf.shape(params)[1]
    preds = tf.reshape(indices, [-1])
    one_hots = tf.expand_dims(tf.one_hot(preds, dim_size), 2)
    return tf.reduce_max(params * one_hots, axis=1)

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
        self.regularization = FLAGS.regularization
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

class Encoder(object):
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

        if initial_state is not None:
            initial_state_fw = initial_state[0]
            initial_state_bw = initial_state[1]
        else:
            initial_state_fw = None
            initial_state_bw = None

        with tf.variable_scope("Encoder"):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_enc, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_enc, forget_bias=1.0)
            hidden_state, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                      lstm_bw_cell,
                                                      embeddings,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sequence_length,
                                                      dtype=tf.float32)
        return tf.concat(hidden_state, 2), final_state

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

    def mix(self, bilstm_encoded_questions, bilstm_encoded_contexts, context_lengths, dropout_placeholder):
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
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_mix, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_mix, forget_bias=1.0)

        # Dimensionalities:
        # D_C_d: samples x context_words x 3*2*n_hidden_enc
        # U: samples x context_words x 2*n_hidden_mix
        C_d_transpose = tf.transpose(coattention_context_C_d, perm = [0, 2, 1])
        D_C_d = tf.concat([bilstm_encoded_contexts, C_d_transpose], 2)
        D_C_d_drop = tf.nn.dropout(D_C_d, dropout_placeholder)

        with tf.variable_scope("Mixer"):
            U, U_final = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, D_C_d_drop, sequence_length=context_lengths, dtype=tf.float32)

        # This gets the final forward & backward hidden states from the output, and concatenates them
        U_final_hidden = tf.concat((U_final[0].h, U_final[1].h), 1)
        return tf.concat(U, 2), U_final_hidden

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

        if self.config.model == "baseline-v5":
            USE_DECODER_VERSION = 5
        elif self.config.model == "baseline-v4":
            USE_DECODER_VERSION = 4
        elif self.config.model == "baseline-v3":
            USE_DECODER_VERSION = 3
        else:
            USE_DECODER_VERSION = 2
        logging.info("Using decoder version %d" % USE_DECODER_VERSION)


        if USE_DECODER_VERSION == 5:
            # Similar to V4, but also pass a one-hot for the start prediction to the end predictor
            weights_shape = (self.config.n_hidden_dec_base * 2, 1)
            
            #bnew_shape = (1)

            #Ureshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])

            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("StartPredictor"):
                with tf.variable_scope("DecoderBiLSTM"):
                    # Forward direction cell
                    decoder_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    # Backward direction cell
                    decoder_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    coattention_encoding_drop = tf.nn.dropout(coattention_encoding, dropout_placeholder)
                    decoder_bilstm_output, _, = tf.nn.bidirectional_dynamic_rnn(decoder_lstm_fw_cell, decoder_lstm_bw_cell, coattention_encoding_drop,
                        sequence_length=context_lengths, dtype=tf.float32)
                    decoder_bilstm_output = tf.concat(decoder_bilstm_output, 2)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(decoder_bilstm_output, [-1, 2 * self.config.n_hidden_dec_base])
                start_pred_tmp = tf.matmul(decoder_bilstm_output_reshape, weights)# + bnew
                start_pred_tmp2 = tf.reshape(start_pred_tmp, [-1, self.config.output_size])
                start_pred = start_pred_tmp2
                #self.start_pred = start_pred # Needed for

            with tf.variable_scope("EndPredictor"):
                with tf.variable_scope("DecoderBiLSTM"):
                    # Forward direction cell
                    decoder_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    # Backward direction cell
                    decoder_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    
                    # Create one hots for the start prediction
                    start_pred_answer = tf.argmax(start_pred, 1) # samples x 1
                    start_pred_answer_expand = tf.expand_dims(start_pred_answer, 1) # samples x 1 x 1
                    start_pred_answer_onehot = tf.one_hot(start_pred_answer_expand, self.config.output_size, axis = 1) # samples x context_words x 1

                    # Concatenate them to the coattention matrix
                    coattention_plus_start_pred = tf.concat((coattention_encoding, start_pred_answer_onehot), 2) # samples x context_words x (2*n_hidden_mix+1)

                    coattention_plus_start_pred_drop = tf.nn.dropout(coattention_plus_start_pred, dropout_placeholder)
                    decoder_bilstm_output, _, = tf.nn.bidirectional_dynamic_rnn(decoder_lstm_fw_cell, decoder_lstm_bw_cell, coattention_plus_start_pred_drop,
                        sequence_length=context_lengths, dtype=tf.float32)
                    decoder_bilstm_output = tf.concat(decoder_bilstm_output, 2)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(decoder_bilstm_output, [-1, 2 * self.config.n_hidden_dec_base])
                end_pred_tmp = tf.matmul(decoder_bilstm_output_reshape, weights)# + bnew
                end_pred_tmp2 = tf.reshape(end_pred_tmp, [-1, self.config.output_size])
                end_pred = end_pred_tmp2# + bnew


        elif USE_DECODER_VERSION == 4:
            # Similar to BiDAF paper
            # For start_pred, just use a vector (like V2)
            # For end_pred, adding an additional BiLSTM

            weights_shape = (self.config.n_hidden_dec_base * 2, 1)
            #bnew_shape = (1)

            #Ureshape = tf.reshape(U, [-1, 2 * self.config.n_hidden_mix])

            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("StartPredictor"):
                with tf.variable_scope("DecoderBiLSTM"):
                    # Forward direction cell
                    decoder_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    # Backward direction cell
                    decoder_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    coattention_encoding_drop = tf.nn.dropout(coattention_encoding, dropout_placeholder)
                    decoder_bilstm_output, _, = tf.nn.bidirectional_dynamic_rnn(decoder_lstm_fw_cell, decoder_lstm_bw_cell, coattention_encoding_drop,
                        sequence_length=context_lengths, dtype=tf.float32)
                    decoder_bilstm_output = tf.concat(decoder_bilstm_output, 2)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(decoder_bilstm_output, [-1, 2 * self.config.n_hidden_dec_base])
                start_pred_tmp = tf.matmul(decoder_bilstm_output_reshape, weights)# + bnew
                start_pred_tmp2 = tf.reshape(start_pred_tmp, [-1, self.config.output_size])
                start_pred = start_pred_tmp2# samples x context_words

            with tf.variable_scope("EndPredictor"):
                with tf.variable_scope("DecoderBiLSTM"):
                    # Forward direction cell
                    decoder_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    # Backward direction cell
                    decoder_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_dec_base, forget_bias=1.0)
                    coattention_encoding_drop = tf.nn.dropout(coattention_encoding, dropout_placeholder)
                    decoder_bilstm_output, _, = tf.nn.bidirectional_dynamic_rnn(decoder_lstm_fw_cell, decoder_lstm_bw_cell, coattention_encoding_drop,
                        sequence_length=context_lengths, dtype=tf.float32)
                    decoder_bilstm_output = tf.concat(decoder_bilstm_output, 2)
                weights = tf.get_variable('weights', shape=weights_shape, initializer=initializer, dtype=tf.float32)
                #bnew = tf.Variable(tf.zeros(bnew_shape, tf.float32))
                decoder_bilstm_output_reshape = tf.reshape(decoder_bilstm_output, [-1, 2 * self.config.n_hidden_dec_base])
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
                end_pred = tf.reshape(end_pred_tmp, [-1, self.config.output_size])


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

    def decode(self, U, unused, context_lengths, dropout):
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

        max_context_length = U.get_shape().as_list()[1]
        embedding_size = U.get_shape().as_list()[2]

        maxout_size = self.config.maxout_size
        max_decode_steps = self.config.max_decode_steps

        highway_alpha = highway_maxout(embedding_size, maxout_size)
        highway_beta = highway_maxout(embedding_size, maxout_size)

        cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)

        # u_s the embeddings of the start guess
        # u_e the embeddings of the end guess
        u_s = U[:,0,:]
        u_e = U[:,0,:]

        # set up initial state
        h = None

        with tf.variable_scope("Decoder") as scope:
            for step in range(max_decode_steps):
                if step > 0: scope.reuse_variables()
                _, h = tf.contrib.rnn.static_rnn(cell, [tf.concat([u_s, u_e], 1)], initial_state=h, dtype=tf.float32)
                h_add = h[0] + h[1]

                with tf.variable_scope('highway_alpha'):
                    alpha = highway_alpha(U, h_add, u_s, u_e)
                    start_preds = tf.argmax(alpha, axis=1)
                    u_s = batch_slice(U, start_preds)

                with tf.variable_scope('highway_beta'):
                    beta = highway_beta(U, h_add, u_s, u_e)
                    end_preds = tf.argmax(beta, axis=1)
                    u_e = batch_slice(U, end_preds)

        return tf.reshape(alpha, [-1, max_context_length]), tf.reshape(beta, [-1, max_context_length])

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
        if GOOGLE3:
            self.pretrained_embeddings = np.load(gfile.GFile(embed_path))["glove"]
        else:
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
            self.setup_loss()
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

        U, U_final = self.mixer.mix(bilstm_encoded_questions, bilstm_encoded_contexts, self.context_lengths_placeholder, self.dropout_placeholder)
        self.start_prediction, self.end_prediction = self.decoder.decode(U, U_final, self.context_lengths_placeholder, self.dropout_placeholder)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        mask = lengths_to_masks(self.context_lengths_placeholder, self.config.output_size)

        start_loss = masked_loss(self.start_prediction, self.answers_numeric_list[:, 0], mask)
        end_loss = masked_loss(self.end_prediction, self.answers_numeric_list[:, 1], mask)

        L2_factor = self.config.regularization
        L2_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if 'weight' in tensor.name ]) * L2_factor

        self.loss = start_loss + end_loss + L2_loss

    def setup_embeddings(self):
        # jorisvanmens: looks up embeddings (code by Ilya)
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            embeddings = tf.constant(self.pretrained_embeddings, dtype=tf.float32)
            question_embeddings_lookup_nodrop = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            context_embeddings_lookup_nodrop = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            # Apply dropout to lookups
            self.question_embeddings_lookup = tf.nn.dropout(question_embeddings_lookup_nodrop, self.dropout_placeholder)
            self.context_embeddings_lookup = tf.nn.dropout(context_embeddings_lookup_nodrop, self.dropout_placeholder)


    def setup_train_op(self):
        optimizer = get_optimizer(self.config.optimizer)
        self.train_op = optimizer(self.config.learning_rate).minimize(self.loss)

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return: loss - the training loss
        """
        input_feed = self.create_feed_dict(train_x, train_y, self.config.dropout)

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
        losses = []

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
                losses.append(loss)
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

            make_prediction_plot(losses, self.config.batch_size, epoch)

            logging.info("Evaluating current model..")
            _, _, valid_loss = self.evaluate_answer(session, dataset['val'], len(dataset['val']))
            logging.info("Validation loss: %s" % format(valid_loss, '.5f'))
            _, _, valid_loss = self.evaluate_answer(session, dataset['train'], len(dataset['val'])) #subset of full dataset for speed
            logging.info("Train loss: %s" % format(valid_loss, '.5f'))
