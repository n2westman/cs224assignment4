import random
import tensorflow as tf

from qa_model import lengths_to_masks, batch_slice, masked_loss

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

    def test_mask_loss(self):
        with self.test_session() as sess:
            masks  = [[1., 1., 1., 1., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 0., 0., 0.]]

            logits = [[100.,   0.,   0.,   0., 500., 500., 500., 500.],
                      [  0., 100.,   0.,   0.,   0.,   0.,   0.,   0.],
                      [  0.,   0., 100., 500., 500., 500., 500., 500.],
                      [  0.,   0.,   0., 100., 500., 500., 500., 500.],
                      [  0.,   0.,   0.,   0., 100., 500., 500., 500.]]

            labels = [0, 1, 2, 3, 4]
            self.assertAllClose(sess.run(masked_loss(logits, labels, masks)), 0.)

    def test_batch_slices_constant(self):
        with self.test_session() as sess:
            params =   [[[1, 7], [2, 8], [3, 9], [4, 1]],
                        [[6, 3], [7, 4], [8, 5], [9, 6]],
                        [[2, 8], [3, 9], [4, 1], [5, 2]]]

            indices = [3, 0, 1]

            expected_out = [[4, 1], [6, 3], [3, 9]]

            self.assertAllClose(sess.run(batch_slice(params, indices)), expected_out)

    def test_batch_slices_generate(self):
        with self.test_session() as sess:
            embedding_size = 3
            max_context_length = 100
            batch_size = 10
            params = [[[j + 100 * k for _ in range(embedding_size)] for j in range(max_context_length)] for k in range(batch_size)]

            indices = [int(random.uniform(0, batch_size)) for _ in range(batch_size)]
            expected_out = [[i + 100 * j] * embedding_size for j, i in enumerate(indices)]

            self.assertAllClose(sess.run(batch_slice(params, indices)), expected_out)


if __name__ == '__main__':
  tf.test.main()
