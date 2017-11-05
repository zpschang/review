import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops
import numpy as np

# Affect-LM model

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

class Aspect_LM_Model():
    def __init__(self,
                vocab_size,
                embedding_size = 128,
                rating_num = 6,
                aspect_num = 6,
                lstm_size = 200,
                num_layer = 4,
                max_length = 15,
                prefix_length = 3,
                max_gradient_norm = 2,
                batch_size = 20,
                learning_rate = 0.001,
                beam_width = 5,

                embed=None):
        print 'building model'

        self.batch_size = batch_size
        self.prefix_length = prefix_length
        self.max_length = max_length

        rating_embedsize = 100
        aspect_embedsize = 100

        with tf.variable_scope("Aspect-LM") as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.ground_truth = tf.placeholder(tf.int32, [None, max_length], "ground_truth")
            self.target_weight = tf.placeholder(tf.float32, [None, None], "target_weight")

            self.prefix = tf.placeholder(tf.int32, [None, None], "prefix") # for inferring
            self.aspect = tf.placeholder(tf.int32, [None], "aspect")
            self.rating = tf.placeholder(tf.int32, [None], "rating")
            # self.beta = tf.placeholder(tf.float32, [], "beta")
            no_rating = tf.cast(tf.equal(self.rating, -1), tf.int32)

            batch_tensor = tf.shape(self.target_weight)[0]
            prefix_batch_tensor = tf.shape(self.prefix)[0]
            prefix_tensor = tf.shape(self.prefix)[1]

            self.input = tf.concat([tf.ones([batch_tensor, 1], dtype=tf.int32) * GO_ID, self.ground_truth[:, :-1]],
                                           axis=1)


            embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
            input_embedded = tf.nn.embedding_lookup(embedding, self.input)

            rating_embedding = tf.get_variable('rating_embedding', [rating_num, rating_embedsize], dtype=tf.float32)
            rating_embedded = tf.nn.embedding_lookup(rating_embedding, self.rating+no_rating)

            aspect_embedding = tf.get_variable('aspect_embedding', [aspect_num, aspect_embedsize], dtype=tf.float32)
            aspect_embedded = tf.nn.embedding_lookup(aspect_embedding, self.aspect+2)

            def single_cell():
                return rnn.BasicLSTMCell(lstm_size)
            def multi_cell():
                return rnn.MultiRNNCell([single_cell() for _ in range(num_layer)])
            cell = multi_cell()
            current_state = cell.zero_state(batch_tensor, tf.float32)

            outputs_train = []
            for time_step in range(max_length):
                with tf.variable_scope('cell_model') as cell_scope:
                    if time_step > 0:
                        cell_scope.reuse_variables()
                    input_tensor = input_embedded[:, time_step, :]
                    input_tensor = tf.concat([input_tensor, rating_embedded, aspect_embedded], axis=1)
                    output, current_state = cell(input_tensor, current_state)
                    output = core.dense(output, vocab_size)
                    outputs_train.append(tf.reshape(output, [batch_tensor, 1, vocab_size]))


            logits = tf.concat(outputs_train, axis=1, name="final_output")
            self.result = tf.argmax(logits, 2, name="output")
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=logits)
            self.loss = tf.reduce_sum(cross_entropy * self.target_weight) / tf.cast(batch_tensor, tf.float32)
            self.perplexity = tf.exp(tf.reduce_sum(cross_entropy * self.target_weight) / tf.reduce_sum(self.target_weight))
            params = scope.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

            # TODO: can be replaced with tf.while_loop

            prefix_go = tf.concat([tf.ones([prefix_batch_tensor, 1], dtype=tf.int32) * GO_ID, self.prefix],
                                   axis=1)
            prefix_embedding = tf.nn.embedding_lookup(embedding, prefix_go)
            rating_embedded_copy = tf.reshape(rating_embedded, [batch_tensor, 1, rating_embedsize]) * tf.ones([batch_tensor, prefix_tensor+1, rating_embedsize])
            aspect_embedded_copy = tf.reshape(aspect_embedded, [batch_tensor, 1, aspect_embedsize]) * tf.ones([batch_tensor, prefix_tensor+1, aspect_embedsize])
            prefix_embedding = tf.concat([prefix_embedding, rating_embedded_copy, aspect_embedded_copy], axis=2)

            _, current_state = tf.nn.dynamic_rnn(cell, prefix_embedding, dtype=tf.float32)
            self.current_state = current_state
            outputs_infer = []

            input_tensor = tf.nn.embedding_lookup(embedding, self.prefix[:, -1])
            for time_step in range(max_length):
                with tf.variable_scope('cell_model', reuse=True) as cell_scope:
                    input_tensor = tf.concat([input_tensor, rating_embedded, aspect_embedded], axis=1)
                    output, current_state = cell(input_tensor, current_state)
                    output = core.dense(output, vocab_size)
                    output = output - 10 * tf.one_hot(2, vocab_size)
                    output_voc = tf.argmax(output, axis=1)
                    input_tensor = tf.nn.embedding_lookup(embedding, output_voc)
                    outputs_infer.append(tf.reshape(output_voc, [batch_tensor, 1]))
            self.result_infer = tf.concat(outputs_infer, axis=1, name="output_infer")

            tf.summary.scalar('perplexity_train', self.perplexity)
            tf.summary.scalar('loss_train', self.loss)
            self.summaries = tf.summary.merge_all()

        print 'build finished'

    def all_params(self):
        with tf.variable_scope('Aspect-LM') as scope:
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

    def process_batch(self, batch, reader, max_length):
        feed_truth = []
        feed_weight = []
        feed_rating = []
        feed_aspect = []

        for comment, rating, aspect in batch:
            feed_rating.append(rating)
            feed_aspect.append(aspect)
            pad_num = max_length - len(comment)
            weight = [1.0] * len(comment) + [0.0] * pad_num
            for index in range(len(comment)):
                if comment[index] == UNK_ID:
                    weight[index] = 0

            comment = comment + [PAD_ID] * pad_num
            if len(comment) > max_length:
                weight = weight[:max_length]
                comment = comment[:max_length]
            feed_truth.append(comment)
            feed_weight.append(weight)

        return feed_truth, feed_weight, feed_rating, feed_aspect

    def update(self, sess, reader):
        batch = reader.get_batch(self.batch_size)
        if batch == None:
            print 'batch is None!!!'
            reader.reset()
            batch = reader.get_batch(self.batch_size)

        # build feed dict
        feed_truth, feed_weight, feed_rating, feed_aspect = self.process_batch(batch, reader, self.max_length)
        feed_dict = {}
        feed_dict[self.ground_truth] = np.array(feed_truth, dtype=np.int32)
        feed_dict[self.target_weight] = np.array(feed_weight, dtype=np.float32)
        feed_dict[self.rating] = feed_rating
        feed_dict[self.aspect] = feed_aspect

        feed_output = [self.result, self.loss, self.perplexity, self.summaries, self.train]
        result, loss, perplexity, summaries, _ = sess.run(feed_output, feed_dict=feed_dict)

        # reader.output(result, batch)
        print 'perplexity = ' + str(perplexity)
        return summaries
    
    def inference(self, sess, prefix_size, reader, file=None):

        batch = reader.get_batch(self.batch_size, prefix_size)

        feed_prefix, feed_weight, feed_rating, feed_aspect = self.process_batch(batch, reader, prefix_size)

        feed_dict = {}
        feed_dict[self.prefix] = np.array(feed_prefix, dtype=np.int32)
        feed_dict[self.target_weight] = np.array(feed_weight, dtype=np.float32)
        feed_dict[self.rating] = feed_rating
        feed_dict[self.aspect] = feed_aspect

        result_infer = sess.run(self.result_infer, feed_dict=feed_dict)

        result_prefix = [tmp[0] for tmp in batch]
        reader.output(result=[np.concatenate([prefix, infer]) for prefix, infer in zip(result_prefix, result_infer)], file=file)

        return result_infer

    def train_state(self, sess, prefix_size, reader):
        batch = reader.get_batch(self.batch_size, prefix_size)

        feed_truth, feed_weight, feed_rating, feed_aspect = self.process_batch(batch, reader)
        feed_dict = {}
        feed_dict[self.prefix] = feed_truth
        feed_dict[self.target_weight] = feed_weight
        feed_dict[self.rating] = feed_rating
        feed_dict[self.aspect] = feed_aspect

if __name__ == "__main__":
    model = Aspect_LM_Model(30000)
    model.all_params()