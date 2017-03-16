from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random
import os

import numpy as np
import tensorflow as tf

from data_utils import shuffle_and_open_dataset, split_in_batches
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
        self.shuffle = FLAGS.shuffle
        self.evaluate = FLAGS.evaluate
        self.learning_rate = FLAGS.learning_rate
        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.dropout = FLAGS.dropout
        self.batch_size = FLAGS.batch_size
        self.epochs = FLAGS.epochs
        self.state_size = FLAGS.state_size
        self.output_size = FLAGS.output_size
        self.embedding_size = FLAGS.embedding_size
        self.n_hidden_mix = FLAGS.n_hidden_mix
        self.n_hidden_dec_base = FLAGS.n_hidden_dec_base
        self.n_hidden_dec_hmn = FLAGS.n_hidden_dec_hmn
        self.max_examples = FLAGS.max_examples
        self.max_question_length = FLAGS.max_question_length
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
        self.n_hidden_enc = FLAGS.n_hidden_enc

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

        if initial_state is not None:
            initial_state_fw = initial_state[0]
            initial_state_bw = initial_state[1]
        else:
            initial_state_fw = None
            initial_state_bw = None

        with tf.variable_scope("Encoder"):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_enc, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_enc, forget_bias=1.0)
            embeddings_drop = tf.nn.dropout(embeddings, self.config.dropout)
            (hidden_state_fw, hidden_state_bw), final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                      lstm_bw_cell,
                                                      embeddings_drop,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sequence_length,
                                                      dtype=tf.float32)
        return hidden_state_fw + hidden_state_bw, final_state

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

        W1 = tf.get_variable('W1', shape=(self.input_size, self.hidden_size), initializer=initializer, dtype=tf.float32)
        b1 = tf.Variable(tf.zeros((1, self.hidden_size), tf.float32))
        W2 = tf.get_variable('W2', shape=(self.hidden_size, self.output_size), initializer=initializer, dtype=tf.float32)
        if self.output_size > 1: # don't need bias if output_size == 1
            b2 = tf.Variable(tf.zeros((1, self.output_size), tf.float32))

        h = tf.nn.relu(tf.matmul(inputs, W1) + b1) # samples x n_hidden_dec
        h_drop = tf.nn.dropout(h, dropout_placeholder)
        if self.output_size > 1:
            output = tf.matmul(h_drop, W2) + b2
        else:
            output = tf.matmul(h_drop, W2) # don't need bias if output_size == 1
        return output # samples x context_words

class QASystem(object):
    def __init__(self, encoder, embed_path, config, model="baseline"):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.config = config
        self.pretrained_embeddings = np.load(embed_path)["glove"]
        self.model = model

        # ==== set up placeholder tokens ========

        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.max_question_length))
        self.questions_lengths_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size))
        self.context_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.output_size))
        self.context_lengths_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size))
        self.answers_numeric_list = tf.placeholder(tf.int32, shape=(self.config.batch_size, 2))
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

        with tf.variable_scope("qindep"):
            bilstm_encoded_questions_reshape = tf.reshape(bilstm_encoded_questions, [-1, self.config.n_hidden_enc])

            q_indep_ffnn = FFNN(self.config.n_hidden_enc, 1, self.config.n_hidden_dec_base)
            q_indep_scores = q_indep_ffnn.forward_prop(bilstm_encoded_questions_reshape, self.dropout_placeholder)

            q_indep_scores = tf.reshape(q_indep_scores, [-1, self.config.max_question_length])
            q_indep_scores = tf.nn.softmax(q_indep_scores)

            q_indep_repr = tf.reduce_sum(tf.expand_dims(q_indep_scores, -1) * bilstm_encoded_questions, axis=1)
            q_indep_repr = tf.tile(tf.expand_dims(q_indep_repr, 1), [1, self.config.output_size, 1])

        with tf.variable_scope("qalign"):
            question_embeddings_reshape = tf.reshape(self.question_embeddings_lookup, [-1, self.config.embedding_size])
            context_embeddings_reshape = tf.reshape(self.context_embeddings_lookup, [-1, self.config.embedding_size])

            with tf.variable_scope("qscore"):
                q_ffnn = FFNN(self.config.embedding_size, 1, self.config.n_hidden_dec_base)
                q_scores = q_ffnn.forward_prop(question_embeddings_reshape, self.dropout_placeholder)
                q_scores = tf.expand_dims(tf.reshape(q_scores, [-1, self.config.max_question_length]), -1)

            with tf.variable_scope("cscore"):
                c_ffnn = FFNN(self.config.embedding_size, 1, self.config.n_hidden_dec_base)
                c_scores = c_ffnn.forward_prop(context_embeddings_reshape, self.dropout_placeholder)
                c_scores = tf.expand_dims(tf.reshape(c_scores, [-1, self.config.output_size]), -1)

            scores = tf.matmul(c_scores, q_scores, transpose_b=True) # question length x context length
            scores = tf.nn.softmax(scores, 1)

            q_align = tf.expand_dims(self.question_embeddings_lookup, 1) * tf.expand_dims(scores, -1)
            q_align = tf.reduce_sum(q_align, 2)

        logging.debug(self.context_embeddings_lookup)
        logging.debug(q_indep_repr)
        logging.debug(q_align)

        final_encoder = BiLSTMEncoder(self.config)
        encoder_inputs = tf.concat([self.context_embeddings_lookup, q_indep_repr, q_align], 2)

        states, _ = final_encoder.encode(encoder_inputs, self.context_lengths_placeholder)

        with tf.variable_scope("start"):
            start_inputs = tf.reshape(states, [-1, self.config.n_hidden_enc])
            output_ffnn = FFNN(self.config.n_hidden_enc, 1, self.config.n_hidden_dec_base)
            start_scores = output_ffnn.forward_prop(start_inputs, self.dropout_placeholder)
            self.start_prediction = tf.reshape(start_scores, [-1, self.config.output_size])

        with tf.variable_scope("end"):
            end_inputs = tf.reshape(states, [-1, self.config.n_hidden_enc])
            output_ffnn = FFNN(self.config.n_hidden_enc, 1, self.config.n_hidden_dec_base)
            end_scores = output_ffnn.forward_prop(end_inputs, self.dropout_placeholder)
            self.end_prediction = tf.reshape(start_scores, [-1, self.config.output_size])

        # spans_list = []
        #
        # for idx in xrange(self.config.output_size):
        #     prefix = tf.tile(tf.expand_dims(states[:,idx,:], 1), [1, self.config.output_size, 1])
        #     spans_list.append(tf.concat([prefix, states], 2))
        #
        # spans = tf.reshape(tf.stack(spans_list, axis=2), [self.config.batch_size, -1, 6 * self.config.embedding_size])
        #
        # with tf.variable_scope("decode"):
        #     spans_reshaped = tf.reshape(spans, [-1, 6 * self.config.embedding_size])
        #     output_ffnn = FFNN(6 * self.config.embedding_size, 1, self.config.n_hidden_dec_base)
        #     span_scores = output_ffnn.forward_prop(spans_reshaped, self.dropout_placeholder)
        #     span_scores = tf.reshape(span_scores, [self.config.batch_size, self.config.output_size, self.config.output_size])
        #
        # self.start_prediction = tf.nn.softmax(tf.reduce_sum(span_scores, axis=2))
        # self.end_prediction = tf.nn.softmax(tf.reduce_sum(span_scores, axis=1))


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

        self.loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

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

        # cap number of samples
        dataset = dataset[:sample]

        questions, question_lengths, contexts, context_lengths, answers = shuffle_and_open_dataset(dataset, shuffle=self.config.shuffle)
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

        if self.config.evaluate:
            logging.info("Evaluating current model..")
            _, _, valid_loss = self.evaluate_answer(session, dataset['val'], len(dataset['val']))
            logging.info("Validation loss: %s" % format(valid_loss, '.5f'))
            _, _, valid_loss = self.evaluate_answer(session, dataset['train'], len(dataset['val'])) #subset of full dataset for speed
            logging.info("Train loss: %s" % format(valid_loss, '.5f'))
            exit()

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            var_params = variable.get_shape().num_elements()
            total_parameters = total_parameters + var_params
            logging.info("Tensor %s has shape %s with %d parameters" % (variable.name, str(shape), var_params))
        logging.info("%d total parameters" % total_parameters)

        for epoch in xrange(self.config.epochs):
            logging.info("Starting epoch %d", epoch)
            questions, question_lengths, contexts, context_lengths, answers = shuffle_and_open_dataset(dataset['train'], shuffle=self.config.shuffle)
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
                    # _, _, valid_loss = self.evaluate_answer(session, dataset['val'])
                    # logging.info("Sample validation loss: %s" % format(valid_loss, '.5f'))
                    if self.config.test: #test the graph
                        logging.info("Graph successfully executes.")

            logging.info("Evaluating current model..")
            _, _, valid_loss = self.evaluate_answer(session, dataset['val'], len(dataset['val']))
            logging.info("Validation loss: %s" % format(valid_loss, '.5f'))
            _, _, valid_loss = self.evaluate_answer(session, dataset['train'], len(dataset['val'])) #subset of full dataset for speed
            logging.info("Train loss: %s" % format(valid_loss, '.5f'))
