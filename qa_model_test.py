import tensorflow as tf

from qa_model import lengths_to_masks

class SquareTest(tf.test.TestCase):
    def test_lengths_to_masks(self):
        with self.test_session() as sess:
            expected_output = [[1., 0., 0., 0., 0., 0., 0., 0.],
                               [1., 1., 0., 0., 0., 0., 0., 0.],
                               [1., 1., 1., 0., 0., 0., 0., 0.],
                               [1., 1., 1., 1., 0., 0., 0., 0.],
                               [1., 1., 1., 1., 1., 0., 0., 0.]]

            lengths_placeholder = tf.placeholder(tf.int32, shape=(None))

            feed_dict = {
                lengths_placeholder: [1,2,3,4,5]
            }

            output_feed = [lengths_to_masks(lengths_placeholder, 8)]

            out = sess.run(output_feed, feed_dict)

            self.assertAllClose(out[0], expected_output)


if __name__ == '__main__':
  tf.test.main()
