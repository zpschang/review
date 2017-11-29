# this file implement the hierarchical decoder model.
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core
import numpy as np
from model import model
from parameter import hyper

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

class HD_model(model):
    def __init__(self):
        self.batch_size = hyper.batch_size
        self.prefix_length = hyper.prefix_length
        self.max_length = hyper.max_length
        self.scope_name = 'HD-model'

        rating_embedsize = 100
        aspect_embedsize = 100
        with tf.variable_scope(self.scope_name) as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.ground_truth = tf.placeholder(tf.int32, [None, None], "ground_truth")
            self.target_weight = tf.placeholder(tf.float32, [None, None], "target_weight")

            self.aspect = tf.placeholder(tf.int32, [None], "aspect")
            self.rating = tf.placeholder(tf.int32, [None], "rating")
            self.is_infer = tf.placeholder(tf.bool, [], 'is_infer')

            no_rating = tf.cast(tf.equal(self.rating, -1), tf.int32)

            batch_tensor = tf.shape(self.ground_truth)[0]
            length_tensor = tf.shape(self.ground_truth)[1]

            self.input = tf.concat([tf.ones([batch_tensor, 1], dtype=tf.int32) * GO_ID, self.ground_truth],
                                   axis=1)
            rating = self.rating + no_rating
            aspect = self.aspect + 2

            embedding = tf.get_variable("embedding", [hyper.vocab_size, hyper.embedding_size], dtype=tf.float32)
            input_embedded = tf.nn.embedding_lookup(embedding, self.input)

            rating_embedding = tf.get_variable('rating_embedding', [hyper.rating_num, rating_embedsize], dtype=tf.float32)
            rating_embedded = tf.nn.embedding_lookup(rating_embedding, rating)

            aspect_embedding = tf.get_variable('aspect_embedding', [hyper.aspect_num, aspect_embedsize], dtype=tf.float32)
            aspect_embedded = tf.nn.embedding_lookup(aspect_embedding, aspect)

            def single_cell():
                return rnn.BasicLSTMCell(hyper.lstm_size)
            def multi_cell():
                return rnn.MultiRNNCell([single_cell() for _ in range(hyper.num_layer)])
            cell = multi_cell()
            current_state = cell.zero_state(batch_tensor, tf.float32)

            all_rating_weight = tf.Variable(tf.truncated_normal([hyper.num_rating, hyper.lstm_size, hyper.vocab_size]))
            all_rating_bias = tf.Variable(tf.zeros([hyper.num_rating, hyper.vocab_size]))

            all_aspect_weight = tf.Variable(tf.truncated_normal([hyper.num_aspect, hyper.lstm_size, hyper.vocab_size]))
            all_aspect_bias = tf.Variable(tf.zeros([hyper.num_aspect, hyper.vocab_size]))

            result_train = []
            result = tf.zeros([batch_tensor, length_tensor])

            for time_step in range(hyper.max_length):
                with tf.variable_scope('cell_model') as cell_scope:
                    if time_step > 0:
                        cell_scope.reuse_variables()

                    result_index = tf.argmax(result, axis=1)
                    refer_input_tensor = tf.nn.embedding_lookup(embedding, result_index)
                    train_input_tensor = input_embedded[:, tf.minimum(time_step, length_tensor), :]
                    input_tensor = tf.cond(tf.logical_and(self.is_infer, time_step > length_tensor),
                                           lambda: refer_input_tensor, lambda: train_input_tensor)
                    rnn_output, current_state = cell(input_tensor, current_state)
                    # rnn_output: batch * lstm_size
                    # hierarchical decoding
                    output_type = tf.nn.softmax(tf.layers.dense(rnn_output, hyper.num_type)) # batch * num_type
                    rating_weight = tf.gather(all_rating_weight, rating) # batch * lstm_size * vocab_size
                    rating_bias = tf.gather(all_rating_bias, rating) # batch * vocab_size
                    aspect_weight = tf.gather(all_aspect_weight, aspect)
                    aspect_bias = tf.gather(all_aspect_bias, aspect)

                    def iter(_, list):
                        x, w, bias = list
                        return tf.reduce_sum(w * tf.expand_dims(x, axis=-1), axis=0) + bias
                    zeros = tf.zeros([hyper.vocab_size])
                    rating_words = tf.scan(iter, (rnn_output, rating_weight, rating_bias), initializer=zeros) # batch * vocab_size
                    aspect_words = tf.scan(iter, (rnn_output, aspect_weight, aspect_bias), initializer=zeros)
                    bg_words = tf.layers.dense(rnn_output, hyper.vocab_size)
                    result = rating_words * output_type[:, 0:1] + aspect_words * output_type[:, 1:2] + bg_words * output_type[:, 2:]

                    result_train.append(tf.reshape(result, [batch_tensor, 1, hyper.vocab_size]))

            logits = tf.concat(result_train, axis=1)

            self.result = tf.argmax(logits, 2, name="output")
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=logits)
            self.loss = tf.reduce_sum(cross_entropy * self.target_weight) / tf.cast(batch_tensor, tf.float32)
            self.perplexity = tf.exp(
                tf.reduce_sum(cross_entropy * self.target_weight) / tf.reduce_sum(self.target_weight))
            params = scope.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            gradients, _ = tf.clip_by_global_norm(gradients, hyper.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(hyper.learning_rate)
            self.train = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

            tf.summary.scalar('perplexity_train', self.perplexity)
            tf.summary.scalar('loss_train', self.loss)
            self.summaries = tf.summary.merge_all()

if __name__ == '__main__':
    hyper.vocab_size = 10000
    model = HD_model()
    model.all_params()