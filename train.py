from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys
import fileinput

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder, HMNDecoder, Mixer, Config
from os.path import join as pjoin
from pdb import set_trace as t
from itertools import izip
from qa_data import PAD_ID

import logging

logging.basicConfig(level=logging.INFO)

# jorisvanmens: these are prefab flags, we're using some of them, and some we don't (would be good to fix)
tf.app.flags.DEFINE_boolean("test", False, "Test that the graph completes 1 batch.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 600, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("max_examples", sys.maxint, "Number of examples over which to iterate")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("model", "baseline", "Model: baseline or MHN (default: baseline)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    # jorisvanmens: this was prefab code (mostly unaltered)
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def load_and_preprocess_dataset(path, dataset, max_context_length, max_examples):
    """
    Creates a dataset. One datum looks as follows: datum1 = [qustion_ids, question_lengths, contexts, ..] (see dataset def)
    Then the whole dataset is a list of this structure: [(datum1), (datum2), ..]

    TODO: could store pre-processed data in standard Tensorflow format:
    https://www.tensorflow.org/programmers_guide/reading_data#standard_tensorflow_format
    (not sure if beneficial, given loading and preprocessing is fast)

    :return: A dictionary of parallel lists.
    dataset = {
        'questions': [],
        'question_lengths': [],
        'contexts': [],
        'context_lengths': [],
        'answers_numeric_list': [],
    }
    """

    logging.info("Loading dataset: %s " % dataset)
    context_ids_file = os.path.join(path, dataset + ".ids.context")
    question_ids_file = os.path.join(path, dataset + ".ids.question")
    answer_span_file = os.path.join(path, dataset + ".span")

    assert os.path.exists(context_ids_file)
    assert os.path.exists(question_ids_file)
    assert os.path.exists(answer_span_file)

    # Definition of the dataset -- note definition appears in multiple places
    dataset = {
        'questions': [],
        'question_lengths': [],
        'contexts': [],
        'context_lengths': [],
        'answers_numeric_list': [],
    }

    # Parameters
    ADD_PADDING = True # Add padding to make all questions and contexts the same length
    FIXED_CONTEXT_SIZE = True # Remove contexts longer than context_size (I don't think this can be turned off anymore)
    context_size = max_context_length # Only relevant for FIXED_CONTEXT_SIZE
    min_input_length = 3 # Remove questions & contexts smaller than this
    num_examples = 0

    with open(context_ids_file) as context_ids, \
         open(question_ids_file) as question_ids, \
         open(answer_span_file) as answer_spans:
        max_context_length = 0
        max_question_length = 0
        for context, question, answer in izip(context_ids, question_ids, answer_spans):
            num_examples += 1

            # Load raw context, question, answer from file
            context = context.split()
            question = question.split()
            answer = answer.split()

            # Don't use Qs / contexts that are too short
            if len(context) < min_input_length or len(question) < min_input_length:
                continue

            # Don't use malformed answers
            if int(answer[0]) > int(answer[1]):
                continue

            # Don't use answers that end after max_context
            if int(answer[1]) > (FLAGS.output_size - 1):
                continue

            # Trim context variables
            if FIXED_CONTEXT_SIZE:
                max_context_length = context_size
                del context[max_context_length:]

            # Add datum to dataset
            dataset['questions'].append(question)
            dataset['question_lengths'].append(len(question))
            dataset['contexts'].append(context)
            dataset['context_lengths'].append(len(context))
            dataset['answers_numeric_list'].append(answer)

            # Track max question & context lengths for adding padding later on
            if ADD_PADDING:
                if len(question) > max_question_length:
                    max_question_length = len(question)
                if not FIXED_CONTEXT_SIZE:
                    if len(context) > max_context_length:
                        max_context_length = len(context)

            if num_examples >= max_examples:
                break;

    # Add padding
    if ADD_PADDING:
        for question in dataset['questions']:
            question.extend([str(PAD_ID)] * (max_question_length - len(question)))
        for context in dataset['contexts']:
            context.extend([str(PAD_ID)] * (max_context_length - len(context)))

    logging.info("Dataset loaded with %s samples" % num_examples)

    return dataset


def main(_):

    # Mix of pre-fab code and our code
    # First function that is called when running code. Loads data, defines a few things and calls train()

    max_context_length = FLAGS.output_size
    dataset = {
        'train': load_and_preprocess_dataset(FLAGS.data_dir, 'train', max_context_length, max_examples=FLAGS.max_examples),
        'val': load_and_preprocess_dataset(FLAGS.data_dir, 'val', max_context_length, max_examples=FLAGS.max_examples)
    }

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    config = Config(batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate, dropout=FLAGS.dropout)
    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    if FLAGS.model == 'baseline':
        decoder = Decoder(output_size=FLAGS.output_size, batch_size=config.batch_size)
    else:
        decoder = HMNDecoder(output_size=FLAGS.output_size, batch_size=config.batch_size)
    mixer = Mixer()

    qa = QASystem(encoder, decoder, mixer, embed_path, max_context_length, config, FLAGS.model)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    #print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)

        # Kick off actual training
        qa.train(sess, dataset, save_train_dir, test=FLAGS.test)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
