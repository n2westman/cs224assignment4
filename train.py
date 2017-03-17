from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import google3
    GOOGLE3 = True
except ImportError:
    GOOGLE3 = False

if GOOGLE3:
    from google3.experimental.users.ikuleshov.cs224n.qa_model import Encoder, QASystem, Decoder, HMNDecoder, Mixer, Config
    from google3.experimental.users.ikuleshov.cs224n.data_utils import load_and_preprocess_dataset
else:
    from data_utils import load_and_preprocess_dataset
    from qa_model import Encoder, QASystem, Decoder, HMNDecoder, Mixer, Config

import os
import json
import sys
import fileinput
import logging

import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from pdb import set_trace as t
from itertools import izip


logging.basicConfig(level=logging.INFO)

# jorisvanmens: these are prefab flags, we're using some of them, and some we don't (would be good to fix)
tf.app.flags.DEFINE_boolean("test", False, "Test that the graph completes 1 batch.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Don't run training but just evaluate on the evaluation set.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.9, "Fraction of units randomly kept (!) on non-recurrent connections.")
tf.app.flags.DEFINE_float("regularization", 0.0001, "L2 regularization constant.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.") # Not used
tf.app.flags.DEFINE_integer("output_size", 600, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("n_hidden_enc", 100, "Number of nodes in the LSTMs of the encoder.")
tf.app.flags.DEFINE_integer("n_hidden_mix", 100, "Number of nodes in the LSTMs of the mixer.")
tf.app.flags.DEFINE_integer("n_hidden_dec_base", 200, "Number of nodes in the hidden layer of decoder V3.")
tf.app.flags.DEFINE_integer("n_hidden_dec_hmn", 50, "Number of nodes in the hidden layer of the HMN.")
tf.app.flags.DEFINE_integer("max_examples", sys.maxint, "Number of examples over which to iterate")
tf.app.flags.DEFINE_integer("maxout_size", 4, "Maxout size for HMN.")
tf.app.flags.DEFINE_integer("max_decode_steps", 3, "Max decode steps for HMN.")
tf.app.flags.DEFINE_integer("batches_per_save", 100, "Save model after every x batches.")
tf.app.flags.DEFINE_integer("after_each_batch", 50, "Evaluate model after every x batches.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("model", "baseline", "Model: baseline or MHN (default: baseline)")

if not GOOGLE3:
    tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    #Currently always uses baseline, need to fix using save_path = os.path.join(train_dir, self.model)
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
    if not GOOGLE3:
        return train_dir    

    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    # Mix of pre-fab code and our code
    # First function that is called when running code. Loads data, defines a few things and calls train()

    dataset = {
        'train': load_and_preprocess_dataset(FLAGS.data_dir, 'train', FLAGS.output_size, max_examples=FLAGS.max_examples),
        'val': load_and_preprocess_dataset(FLAGS.data_dir, 'val', FLAGS.output_size, max_examples=FLAGS.max_examples)
    }

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    config = Config(FLAGS)
    encoder = Encoder(config)
    if FLAGS.model == 'baseline' or FLAGS.model == 'baseline-v2' or FLAGS.model == 'baseline-v3' or FLAGS.model == 'baseline-v4':
        decoder = Decoder(config)
    else:
        decoder = HMNDecoder(config)
    mixer = Mixer(config)

    qa = QASystem(encoder, decoder, mixer, embed_path, config, FLAGS.model)

    if not GOOGLE3:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

    logging.info("Model parameters: %s" % vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)

        if FLAGS.evaluate:
            logging.info("Evaluating current model..")
            _, _, valid_loss = qa.evaluate_answer(sess, dataset['val'], len(dataset['val']))
            logging.info("Validation loss: %s" % format(valid_loss, '.5f'))
            _, _, valid_loss = qa.evaluate_answer(sess, dataset['train'], len(dataset['val'])) #subset of full dataset for speed
            logging.info("Train loss: %s" % format(valid_loss, '.5f'))
        else:
            # Kick off actual training
            qa.train(sess, dataset, save_train_dir)

if __name__ == "__main__":
    tf.app.run()
