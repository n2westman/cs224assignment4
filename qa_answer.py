from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from data_utils import split_in_batches, process_data, word2chars
from qa_model import Encoder, QASystem, Decoder, HMNDecoder, Mixer, Config
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("test", False, "Test that the graph completes 1 batch.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Don't run training but just evaluate on the evaluation set.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 1.0, "Fraction of units randomly kept (!) on non-recurrent connections.")
tf.app.flags.DEFINE_float("regularization", 0.0001, "L2 regularization constant.")
tf.app.flags.DEFINE_integer("batch_size", 200, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.") # Not used
tf.app.flags.DEFINE_integer("output_size", 600, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("n_hidden_enc", 200, "Number of nodes in the LSTMs of the encoder.")
tf.app.flags.DEFINE_integer("n_hidden_mix", 200, "Number of nodes in the LSTMs of the mixer.")
tf.app.flags.DEFINE_integer("n_hidden_dec_base", 200, "Number of nodes in the hidden layer of decoder V3.")
tf.app.flags.DEFINE_integer("n_hidden_dec_hmn", 50, "Number of nodes in the hidden layer of the HMN.")
tf.app.flags.DEFINE_integer("max_examples", sys.maxint, "Number of examples over which to iterate")
tf.app.flags.DEFINE_integer("maxout_size", 32, "Maxout size for HMN.")
tf.app.flags.DEFINE_integer("max_decode_steps", 4, "Max decode steps for HMN.")
tf.app.flags.DEFINE_integer("batches_per_save", 100, "Save model after every x batches.")
tf.app.flags.DEFINE_integer("after_each_batch", 10, "Evaluate model after every x batches.")
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
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

# Char CNN parameters
tf.app.flags.DEFINE_string("filters_list", "100", "Out channel dims (number of filters), separated by commas")
tf.app.flags.DEFINE_string("kernel_lengths", "5", "Kernel (filter) heights, separated by commas")
tf.app.flags.DEFINE_integer("max_question_length", 30, "Maximum quesion size in words.")
tf.app.flags.DEFINE_integer("max_word_length", 30, "Maximum size of a word in characters.")
tf.app.flags.DEFINE_integer("char_out_size", 100, "Char-CNN output size")
tf.app.flags.DEFINE_integer("char_emb_size", 8, "Character embedding size.")
tf.app.flags.DEFINE_integer("char_vocab_size", 250, "Characters vocab count (should span from min to max odrinal value of each char in dataset.")
tf.app.flags.DEFINE_boolean("use_char_cnn_embedding", False, "Use character level convolutional embedding.")

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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    context_data_chars = []
    query_data = []
    query_data_chars = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                context_chars = [word2chars(w, FLAGS.max_word_length) for w in context_tokens]
    

                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]
                question_chars = [word2chars(w, FLAGS.max_word_length) for w in question_tokens]

                context_data.append(context_ids)
                context_data_chars.append(context_chars)
                query_data.append(qustion_ids)
                query_data_chars.append(question_chars)
                question_uuid_data.append(question_uuid)

    return context_data, query_data, context_data_chars, query_data_chars, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, context_data_chars, query_data_chars, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, context_data_chars, query_data_chars, question_uuid_data


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    contexts, context_lengths, questions, question_lengths, question_tokens, answer_tokens, question_uuids = dataset
    counter = 0

    batches = split_in_batches(questions, question_lengths, contexts, context_lengths, question_tokens, answer_tokens, FLAGS.batch_size, question_uuids=question_uuids)

    for batch_x, batch_uuids in batches:
        counter += 1
        logging.info("Reading batch %s." % counter)
        start_indices, end_indices = model.answer(sess, batch_x)
        for idx in range(len(batch_uuids)):
            context = batch_x['contexts'][idx]
            sentence_ids = context[start_indices[idx]: end_indices[idx] + 1]
            sentence = " ".join(map(lambda x: rev_vocab[int(x)], sentence_ids))
            answers[batch_uuids[idx]] = sentence

    logging.info("Produced %s answers for %s questions." % (len(answers), len(questions)))

    return answers


def get_normalized_train_dir(train_dir):
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

def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    logging.info(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, context_data_chars, question_data_chars, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)

    context_data, context_lengths = process_data(context_data, FLAGS.output_size)
    question_data, question_lengths = process_data(question_data, FLAGS.max_question_length)

    # TODO: use process_data()
    for question_token in question_data_chars:
        question_token.extend([ [qa_data.PAD_ID] * FLAGS.max_word_length] * (FLAGS.max_question_length - len(question_token)))

    for context_token in context_data_chars:
        context_token.extend([ [qa_data.PAD_ID] * FLAGS.max_word_length] * (FLAGS.output_size - len(context_token)))

    dataset = (context_data, context_lengths, question_data, question_lengths, question_data_chars, context_data_chars, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    config = Config(FLAGS)
    encoder = Encoder(config)
    if FLAGS.model == 'baseline' or FLAGS.model == 'baseline-v2' or FLAGS.model == 'baseline-v3' or FLAGS.model == 'baseline-v4' or FLAGS.model == 'baseline-v5':
        decoder = Decoder(config)
    else:
        decoder = HMNDecoder(config)
    mixer = Mixer(config)

    qa = QASystem(encoder, decoder, mixer, embed_path, config, FLAGS.model)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
