import os
import sys
import logging
import random

from os.path import join as pjoin
from itertools import izip
from qa_data import PAD_ID

logging.basicConfig(level=logging.INFO)

def shuffle_and_open_dataset(dataset, shuffle=True):
    if shuffle:
        random.shuffle(dataset)

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
