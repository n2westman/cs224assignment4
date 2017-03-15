import tensorflow as tf

from data_utils import load_and_preprocess_dataset, shuffle_and_open_dataset
from qa_data import PAD_ID

class DataUtilsTest(tf.test.TestCase):
    def test_max_examples(self):
        dataset = load_and_preprocess_dataset('./test', 'test', 4, 4)
        assert len(dataset) == 4
        dataset = load_and_preprocess_dataset('./test', 'test', 4, 3)
        assert len(dataset) == 3

    def test_cap_length(self):
        max_context_length = 5
        dataset = load_and_preprocess_dataset('./test', 'test', max_context_length, 4)
        _, _, contexts, _, _ = shuffle_and_open_dataset(dataset, shuffle=False)

        for context in contexts:
            assert len(context) == max_context_length

    def test_add_context_padding_length(self):
        max_context_length = 5
        dataset = load_and_preprocess_dataset('./test', 'test', max_context_length, 4)
        _, _, contexts, context_lengths, _ = shuffle_and_open_dataset(dataset, shuffle=False)

        for context, length in zip(contexts, context_lengths):
            for val in context[:length]:
                assert val != PAD_ID
            for val in context[length:]:
                assert val == PAD_ID


if __name__ == '__main__':
  tf.test.main()
