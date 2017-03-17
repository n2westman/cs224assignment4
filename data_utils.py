import os
import sys
import logging
import random

import matplotlib
import matplotlib.pyplot as plt

from os.path import join as pjoin
from itertools import izip
from qa_data import PAD_ID

logging.basicConfig(level=logging.INFO)

def open_dataset(dataset):
    inputs, answers = zip(*dataset)
    questions, question_lengths, contexts, context_lengths = zip(*inputs)
    return questions, question_lengths, contexts, context_lengths, answers

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

    assert os.path.exists(context_ids_file)
    assert os.path.exists(question_ids_file)
    assert os.path.exists(answer_span_file)

    # Definition of the dataset -- note definition appears in multiple places
    questions = []
    question_lengths = []
    contexts = []
    context_lengths = []
    answers = []

    # Parameters
    min_input_length = 3 # Remove questions & contexts smaller than this
    num_examples = 0

    with open(context_ids_file) as context_ids, \
         open(question_ids_file) as question_ids, \
         open(answer_span_file) as answer_spans:
        max_question_length = 0
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

            # Trim context variables
            context = context[:max_context_length]

            # Add datum to dataset
            questions.append(question)
            question_lengths.append(len(question))
            contexts.append(context)
            context_lengths.append(len(context))
            answers.append(answer)

            # Track max question & context lengths for adding padding later on
            if len(question) > max_question_length:
                max_question_length = len(question)

            num_examples += 1
            if num_examples >= max_examples:
                break;

    # Add padding
    for question in questions:
        question.extend([str(PAD_ID)] * (max_question_length - len(question)))
    for context in contexts:
        context.extend([str(PAD_ID)] * (max_context_length - len(context)))

    dataset = zip(zip(questions, question_lengths, contexts, context_lengths), answers)

    logging.info("Dataset loaded with %s samples" % len(dataset))
    logging.debug("Max question length: %s" % max_question_length)
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
