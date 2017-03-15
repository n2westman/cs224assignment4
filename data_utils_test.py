import tensorflow as tf
import logging

from data_utils import load_and_preprocess_dataset, shuffle_and_open_dataset, split_in_batches
from qa_data import PAD_ID

logging.disable(logging.INFO)

TEST_DATA_PATH = './test'
TEST_DATA_FILE = 'test'

class DataUtilsTest(tf.test.TestCase):
    def test_max_examples(self):
        dataset = load_and_preprocess_dataset(TEST_DATA_PATH, TEST_DATA_FILE, 4, 4)
        assert len(dataset) == 4
        dataset = load_and_preprocess_dataset(TEST_DATA_PATH, TEST_DATA_FILE, 4, 3)
        assert len(dataset) == 3

    def test_cap_length(self):
        max_context_length = 5
        dataset = load_and_preprocess_dataset(TEST_DATA_PATH, TEST_DATA_FILE, max_context_length, 4)
        _, _, contexts, _, _ = shuffle_and_open_dataset(dataset, shuffle=False)

        for context in contexts:
            assert len(context) == max_context_length

    def test_add_context_padding_length(self):
        max_context_length = 10
        dataset = load_and_preprocess_dataset(TEST_DATA_PATH, TEST_DATA_FILE, max_context_length, 4)
        _, _, contexts, context_lengths, _ = shuffle_and_open_dataset(dataset, shuffle=False)

        for context, length in zip(contexts, context_lengths):
            for val in context[:length]:
                assert int(val) != PAD_ID
            for val in context[length:]:
                assert int(val) == PAD_ID

    def test_split_in_batches(self):
        dataset = load_and_preprocess_dataset(TEST_DATA_PATH, TEST_DATA_FILE, 10, 4)
        dataset = dataset * 10 # artificially increase dataset size

        num_data_points = len(dataset) # 40
        batch_size = 9
        num_batches = 5

        questions, question_lengths, contexts, context_lengths, answers = shuffle_and_open_dataset(dataset, shuffle=False)

        batches = split_in_batches(questions, question_lengths, contexts, context_lengths, batch_size, answers=answers)

        assert len(batches) == num_batches

if __name__ == '__main__':
  tf.test.main()
