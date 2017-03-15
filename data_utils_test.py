import tensorflow as tf

from data_utils import load_and_preprocess_dataset

class DataUtilsTest(tf.test.TestCase):
    def test_max_examples(self):
        dataset = load_and_preprocess_dataset('./test', 'test', 4, 4)
        print len(dataset)
        assert len(dataset) == 4
        dataset = load_and_preprocess_dataset('./test', 'test', 4, 3)
        assert len(dataset) == 3


if __name__ == '__main__':
  tf.test.main()
