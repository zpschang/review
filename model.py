import tensorflow as tf
import numpy as np

class model(): # base class for all models
    def __init__(self):
        pass
    def all_params(self):
        with tf.variable_scope(self.scope_name) as scope:
            total = 0
            for var in scope.trainable_variables():
                shape = var.get_shape()
                k = 1
                print shape,
                for dim in shape:
                    k *= dim.value
                print k, var.name
                total += k
            print 'total:', total

    def update(self, sess, reader):
        batch = reader.get_batch(self.batch_size)
        feed_truth, feed_weight, feed_rating, feed_aspect = reader.process_batch(batch, self.max_length)

        feed_dict = {}
        feed_dict[self.ground_truth] = feed_truth
        feed_dict[self.target_weight] = feed_weight
        feed_dict[self.rating] = feed_rating
        feed_dict[self.aspect] = feed_aspect
        feed_dict[self.is_infer] = False

        feed_output = [self.result, self.loss, self.perplexity, self.summaries, self.train]
        result, loss, perplexity, summaries, _ = sess.run(feed_output, feed_dict=feed_dict)

        print 'perplexity = ' + str(perplexity)
        return summaries

    def inference(self, sess, reader, file=None):
        batch = reader.get_batch(self.batch_size, self.prefix_length)
        feed_truth, feed_weight, feed_rating, feed_aspect = reader.process_batch(batch, self.prefix_length)

        feed_dict = {}
        feed_dict[self.ground_truth] = np.array(feed_truth, dtype=np.int32)
        feed_dict[self.target_weight] = np.array(feed_weight, dtype=np.float32)
        feed_dict[self.rating] = feed_rating
        feed_dict[self.aspect] = feed_aspect
        feed_dict[self.is_infer] = True

        result = sess.run(self.result, feed_dict=feed_dict)

        result = result[:, self.prefix_length:]
        reader.output(batch=batch, result=result, file=file)