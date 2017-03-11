from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from pdb import set_trace as t

from evaluate import exact_match_score, f1_score

from contrib_ops import highway_maxout, batch_linear

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

batch_size = 1
hidden_size = 200
maxout_size = 32
max_timesteps = 600
max_decode_steps = 4

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
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_enc, forget_bias=1.0)
        with tf.variable_scope("QuestionEncoderBiLSTM"):

            question_outputs, _, = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, question_embeddings,
                                                  sequence_length=question_lengths, dtype=tf.float32)

        with tf.variable_scope("AnswerEncoderBiLSTM"):
            context_outputs, _, = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, context_embeddings,
                                                  sequence_length=context_lengths, dtype=tf.float32)
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

        U, U_final = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, D_C_d, sequence_length=context_lengths, dtype=tf.float32)
        # This gets the final forward & backward hidden states from the output, and concatenates them
        U_final_hidden = tf.concat((U_final[0].h, U_final[1].h), 1)
        return tf.concat(U, 2), U_final_hidden

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.n_hidden_dec = 50

    def decode(self, coattention_encoding, coattention_encoding_final_states, context_lengths):
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
        W_shape = (n_hidden_mix, self.n_hidden_dec)
        b1_shape = (1, self.n_hidden_dec)

        # Then we want to get our outputs as a context_words vector
        # We do pred = h * V + b2
        # We create V with dimensions: n_hidden_dec x context_words
        # Also b2 with dimensions context_words
        
        V_shape = (self.n_hidden_dec, max_context_words)
        b2_shape = (1, max_context_words)

        # We want to do this for start and end prediction
        with tf.variable_scope("StartPredictor"):
            self.W = tf.get_variable('W', shape = W_shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            self.b1 = tf.Variable(tf.zeros(b1_shape, tf.float32))
            # UW and UWb1 dimensionality: samples x n_hidden_dec
            UW = tf.matmul(U_final, self.W)
            UWb1 = tf.add(UW, self.b1)
            h = tf.nn.relu(UWb1) # samples x n_hidden_dec
            self.V = tf.get_variable('V', shape = V_shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            self.b2 = tf.Variable(tf.zeros(b2_shape, tf.float32))
            hV = tf.matmul(h, self.V)
            self.start_pred = tf.add(hV, self.b2) # samples x context_words

        with tf.variable_scope("EndPredictor"):
            self.W = tf.get_variable('W', shape = W_shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            self.b1 = tf.Variable(tf.zeros(b1_shape, tf.float32))
            # UW and UWb1 dimensionality: samples x n_hidden_dec
            UW = tf.matmul(U_final, self.W)
            UWb1 = tf.add(UW, self.b1)
            h = tf.nn.relu(UWb1) # samples x n_hidden_dec
            self.V = tf.get_variable('V', shape = V_shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            self.b2 = tf.Variable(tf.zeros(b2_shape, tf.float32))
            hV = tf.matmul(h, self.V)
            self.end_pred = tf.add(hV, self.b2) # samples x context_words
        
        return self.start_pred, self.end_pred


class HMNDecoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.n_hidden_dec = 50

    def decode(self, coattention_encoding, coattention_encoding_final_states, context_lengths):
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
        # coattention_encoding: samples x context_words x 2*n_hidden_mix (it's packed in like this: ((data)), for some reason)
        # return value: samples x context_words x 2*n_hidden_dec
        self._initial_guess = np.zeros((2, batch_size), dtype=np.int32)
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
            loop_until = tf.to_int32(np.array(range(batch_size)))
            # initial estimated positions 
            # s and e have dimension [batch_size]
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
                  s = tf.reshape(tf.argmax(alpha, 0), [batch_size])
                  # update start guess
                  fn = lambda idx: select(self._u, s, idx)
                  u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                with tf.variable_scope('highway_beta'):
                  # compute end position next
                  fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                  beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                  e = tf.reshape(tf.argmax(beta, 0), [batch_size])
                  # update end guess
                  fn = lambda idx: select(self._u, e, idx)
                  u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                self._s.append(s)
                self._e.append(e)
                self._alpha.append(tf.reshape(alpha, [batch_size, -1]))
                self._beta.append(tf.reshape(beta, [batch_size, -1]))
        return self._alpha, self._beta

class QASystem(object):
    def __init__(self, encoder, decoder, mixer, embed_path, max_context_length, model):
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
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, max_context_length))
        self.context_lengths_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.answer_starts_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.answer_ends_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.answers_numeric_list = tf.placeholder(tf.int32, shape=(None, 2))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            question_embeddings_lookup, context_embeddings_lookup = self.setup_embeddings()
            self.setup_system(
                question_embeddings_lookup,
                context_embeddings_lookup
            )
            if model == 'baseline':
                self.setup_loss()
            else:
                self.setup_hmn_loss()
            self.setup_train_op()

        # ==== set up training/updating procedure ====
        pass

    def split_in_batches(self, dataset, batch_size):
        batches = []
        for start_index in range(0, len(dataset['questions']), batch_size):
            batch = {
                'questions': [],
                'question_lengths': [],
                'contexts': [],
                'context_lengths': [], 
                'answer_starts_onehot': [],
                'answer_ends_onehot': [],
                'answers_numeric_list': []
            }
            batch['questions'] = dataset['questions'][start_index:start_index + batch_size]
            batch['question_lengths'] = dataset['question_lengths'][start_index:start_index + batch_size]
            batch['contexts'] = dataset['contexts'][start_index:start_index + batch_size]
            batch['context_lengths'] = dataset['context_lengths'][start_index:start_index + batch_size]
            batch['answer_starts_onehot'] = dataset['answer_starts_onehot'][start_index:start_index + batch_size]
            batch['answer_ends_onehot'] = dataset['answer_ends_onehot'][start_index:start_index + batch_size]
            batch['answers_numeric_list'] = dataset['answers_numeric_list'][start_index:start_index + batch_size]
            batches.append(batch)

        print("Created", str(len(batches)), "batches")
        return batches


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

        self.coattention_encoding, self.coattention_encoding_final_states = self.mixer.mix(bilstm_encoded_questions, bilstm_encoded_contexts, self.context_lengths_placeholder)
        self.start_prediction, self.end_prediction = self.decoder.decode(self.coattention_encoding, self.coattention_encoding_final_states, self.context_lengths_placeholder)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        # jorisvanmens: This is not really tested, although it does seem to work
        sm_ce_loss_answer_start = tf.nn.softmax_cross_entropy_with_logits(logits = self.start_prediction, labels = self.answer_starts_placeholder)
        sm_ce_loss_answer_end = tf.nn.softmax_cross_entropy_with_logits(logits = self.end_prediction, labels = self.answer_ends_placeholder)
        self.loss = tf.reduce_mean(sm_ce_loss_answer_start) + tf.reduce_mean(sm_ce_loss_answer_end)

    def setup_hmn_loss(self):
        def _loss_shared(logits, labels):
          labels = tf.reshape(labels, [batch_size])
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

        alpha_true = self.answer_starts_placeholder
        beta_true = self.answer_ends_placeholder
        self.loss = _loss_multitask(self.decoder._alpha, alpha_true,
                                    self.decoder._beta, beta_true)


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            question_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            context_embeddings_lookup = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            return question_embeddings_lookup, context_embeddings_lookup

    def setup_train_op(self):
        learning_rate = 0.5
        optimizer = get_optimizer("adam")
        #optimizer = tf.train.AdamOptimizer(0.5)
        self.train_op = optimizer().minimize(self.loss)
        return self.train_op

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

    def evaluate_answer(self, session, data_batches, sample=100, log=True):
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

        # jorisvanmens: I built this function from scratch (not conform original "specification")

        f1 = 0.
        em = 0.

        test_batch = random.choice(data_batches)
        feed_dict = self.prep_feed_dict_from_batch(test_batch)
        answer_start_predictions, answer_end_predictions, answers_numeric_list = \
            session.run([self.start_prediction, self.end_prediction, self.answers_numeric_list], feed_dict)
        
        answer_start_predictions_numeric = np.argmax(answer_start_predictions, axis = 1)
        answer_end_predictions_numeric = np.argmax(answer_end_predictions, axis = 1)
        f1s = []
        ems = []

        for idx, answer_numeric in enumerate(answers_numeric_list):
            prediction = [answer_start_predictions_numeric[idx], answer_end_predictions_numeric[idx]]
            em = 0
            if prediction[0] == answer_numeric[0] and prediction[1] == answer_numeric[1]:
                em = 1
            f1 = 0
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

    def prep_feed_dict_from_batch(self, batch):
        feed_dict = {
            self.question_placeholder: batch['questions'],
            self.questions_lengths_placeholder: batch['question_lengths'],
            self.context_placeholder: batch['contexts'],
            self.context_lengths_placeholder: batch['context_lengths'],
            self.answer_starts_placeholder: batch['answer_starts_onehot'],
            self.answer_ends_placeholder: batch['answer_ends_onehot'],
            self.answers_numeric_list: batch['answers_numeric_list']
        }
        return feed_dict


    def test_the_graph(self, session, dataset):

        batch_size = 100
        data_batches = self.split_in_batches(dataset, batch_size)
        data_input = data_batches[0]

        prep_feed_dict_from_batch(data_input)

        # Dimensionalities:
        # question_placeholder: samples x words
        # context_placeholder: samples x words
        # question_embeddings_lookup: samples x words x embed_size
        # context_embeddings_lookup: samples x words x embed_size
        # bilstm_encoded_questions: samples x words x 2*n_hidden_enc
        # bilstm_encoded_contexts: samples x words x 2*n_hidden_enc

        # print_debug_output = tf.Print(self.network, [self.network], summarize=500)

        out1 = session.run([self.loss], feed_dict)
        print("Final layer shape:", out1[0].shape)
        print("Loss: ", out1[0])
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

        batch_size = 100
        evaluate_after_batches = 10 # Note one evaluation takes as much time
        data_batches = self.split_in_batches(dataset, batch_size)

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for idx, batch in enumerate(data_batches):
            tic = time.time()
            feed_dict = self.prep_feed_dict_from_batch(batch)
            _, current_loss = session.run([self.train_op, self.loss], feed_dict)
            toc = time.time()
            print("Batch", str(idx), "done with", current_loss, "loss (took", str(format(toc - tic, '.2f')), "seconds)")
            if (idx + 1) % evaluate_after_batches == 0:
                f1, em = self.evaluate_answer(session, data_batches)
                print("F1:", f1, " EM:", em)

