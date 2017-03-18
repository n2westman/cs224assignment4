try:
    import google3
    GOOGLE3 = True
except ImportError:
    GOOGLE3 = False

if GOOGLE3:
    from google3.experimental.users.ikuleshov.cs224n.qa_data import PAD_ID
else:
    from qa_data import PAD_ID

import os
import sys
import logging
import random
import tensorflow as  tf

import matplotlib
import matplotlib.pyplot as plt

from os.path import join as pjoin
from itertools import izip

logging.basicConfig(level=logging.INFO)

def open_dataset(dataset):
    inputs, answers = zip(*dataset)
    questions, question_lengths, contexts, context_lengths = zip(*inputs)
    return questions, question_lengths, contexts, context_lengths, answers

def process_data(data_list, max_length=None):
    """
    Processes a list of contexts or questions.

    This will:
    (1) Trim each sequence to max_length, if specified.
    (2) Obtain the lengths of each sequence (up to max_length)
    (3) Pad the sequence up to the max length.

    """

    if max_length is None:
        max_length = max(map(len, data_list))

    data_list = cap_sequences(data_list, max_length)
    lengths = [len(x) for x in data_list]

    pad_sequences(data_list, max_length)

    return data_list, lengths

def cap_sequences(sequences, max_length):
    """
    Trims a batch of sequences to a specified max_length.
    """
    return [x[:max_length] for x in sequences]

def pad_sequences(sequences, max_length):
    """
    Pads a batch of sequences (in-place) to a specified max length.

    TODO(nwestman): make this functional and not in-place

    :return:
    """
    for sequence in sequences:
        sequence.extend([str(PAD_ID)] * (max_length - len(sequence)))

def load_and_preprocess_dataset(path, dataset, max_context_length, max_examples):
    """
    Creates a dataset. One datum looks as follows: datum1 = [qustion_ids, question_lengths, contexts, ..] (see dataset def)
    Then the whole dataset is a list of this structure: [(datum1), (datum2), ..]

    TODO: could store pre-processed data in standard Tensorflow format:
    https://www.tensorflow.org/programmers_guide/reading_data#standard_tensorflow_format
    (not sure if beneficial, given loading and preprocessing is fast)

    :return: A list of (inputs, labels) tuples, where inputs are (q, c)
    """

    logging.info("Loading dataset: %s " % dataset)
    context_ids_file = os.path.join(path, dataset + ".ids.context")
    question_ids_file = os.path.join(path, dataset + ".ids.question")
    answer_span_file = os.path.join(path, dataset + ".span")

    assert tf.gfile.Exists(context_ids_file)
    assert tf.gfile.Exists(question_ids_file)
    assert tf.gfile.Exists(answer_span_file)

    # Definition of the dataset -- note definition appears in multiple places
    questions = []
    contexts = []
    answers = []

    # Parameters
    min_input_length = 3 # Remove questions & contexts smaller than this
    num_examples = 0

    with tf.gfile.GFile(context_ids_file) as context_ids, \
         tf.gfile.GFile(question_ids_file) as question_ids, \
         tf.gfile.GFile(answer_span_file) as answer_spans:
        for context, question, answer in izip(context_ids, question_ids, answer_spans):
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
            if int(answer[1]) > (max_context_length - 1):
                continue

            # Add datum to dataset
            questions.append(question)
            contexts.append(context)
            answers.append(answer)

            num_examples += 1
            if num_examples >= max_examples:
                break;

    questions, question_lengths = process_data(questions)
    contexts, context_lengths = process_data(contexts, max_context_length)

    dataset = zip(zip(questions, question_lengths, contexts, context_lengths), answers)

    logging.info("Dataset loaded with %s samples" % len(dataset))
    logging.debug("Max question length: %s" % max(question_lengths))
    logging.debug("Max context length: %s" % max_context_length)

    return dataset

def split_in_batches(questions, question_lengths, contexts, context_lengths, batch_size, answers=None, question_uuids=None):
    """
    Splits a dataset into batches, each of batch_size.
    """
    batches = []
    for start_index in range(0, len(questions), batch_size):
        batch_x = {
            'questions': questions[start_index:start_index + batch_size],
            'question_lengths': question_lengths[start_index:start_index + batch_size],
            'contexts': contexts[start_index:start_index + batch_size],
            'context_lengths': context_lengths[start_index:start_index + batch_size],
        }
        if answers is not None:
            batch_y = answers[start_index:start_index + batch_size]
            batches.append((batch_x, batch_y))
        elif question_uuids is not None:
            batch_uuids = question_uuids[start_index:start_index + batch_size]
            batches.append((batch_x, batch_uuids))
        else:
            raise ValueError("Neither answers nor question uuids were provided")

    logging.debug("Expected batch_size: %s" % batch_size)
    logging.debug("Actual batch_size: %s" % len(batches[0][1]))
    if len(questions) > batch_size:
        assert (batch_size == len(batches[0][1]))

    logging.info("Created %d batches" % len(batches))
    return batches

def make_prediction_plot(losses, batch_size, epoch):
    plt.subplot(2, 1, 1)
    plt.title("Losses")
    plt.plot(np.arange(len(losses)), losses, label="Loss")
    plt.ylabel("Loss")

    plt.xlabel("Minibatch (size %s)" % batch_size)
    output_path = "Losses-Epoch-%s.png" % epoch
    plt.savefig(output_path)
