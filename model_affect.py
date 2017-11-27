import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops
import numpy as np
import time

# Affect-LM model

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

class Affect_LM_Model():
    def __init__(self,
                vocab_size,
                feature_size,
                embedding_size = 128,
                lstm_size = 200,
                num_layer = 4,
                max_length = 30,
                prefix_length=3,
                max_gradient_norm = 2,
                batch_size = 20,
                learning_rate = 0.001,
                is_pure_LM = False,
                beam_width = 5,
                embed=None):
        # TODO: unify training/testing inputs
        # TODO: make it compatible with other models
        self.batch_size = batch_size
        self.prefix_length = prefix_length
        self.max_length = max_length
        self.scope_name = 'Affect-LM'

        with tf.variable_scope(self.scope_name) as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.ground_truth = tf.placeholder(tf.int32, [None, max_length], "ground_truth")
            self.target_weight = tf.placeholder(tf.float32, [None, None], "target_weight")
            self.feature = tf.placeholder(tf.float32, [None, max_length, feature_size], "feature") # for training
            self.fixed_feature = tf.placeholder(tf.float32, [None, feature_size], "fixed_feature") # for inferring
            self.prefix = tf.placeholder(tf.int32, [None, None], "prefix") # for inferring
            self.beta = tf.placeholder(tf.float32, [], "beta")

            batch_tensor = tf.shape(self.target_weight)[0]
            prefix_batch_tensor = tf.shape(self.prefix)[0]

            self.input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32) * GO_ID, self.ground_truth[:, :-1]],
                                           axis=1)

            embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
            input_embedding = tf.nn.embedding_lookup(embedding, self.input)

            def single_cell():
                return rnn.BasicLSTMCell(lstm_size)
            def multi_cell():
                return rnn.MultiRNNCell([single_cell() for _ in range(num_layer)])
            cell = multi_cell()
            current_state = cell.zero_state(batch_tensor, tf.float32)

            def cell_model(cell, input_tensor, current_state, feature):
                cell_output, current_state = cell(input_tensor, current_state)

                if not is_pure_LM:
                    feature = feature * self.beta
                    feature_output = tf.nn.relu(core.dense(feature, 100))
                    feature_output = tf.nn.relu(core.dense(feature_output, 200))
                    output = tf.concat([cell_output, feature_output], axis=1)
                    output = core.dense(output, vocab_size)
                else:
                    output = core.dense(cell_output, vocab_size)
                return output, current_state

            outputs_train = []
            for time_step in range(max_length):
                with tf.variable_scope('cell_model') as cell_scope:
                    if time_step > 0:
                        cell_scope.reuse_variables()
                    input_tensor = input_embedding[:, time_step, :]
                    feature = self.feature[:, time_step, :]
                    output, current_state = cell_model(cell, input_tensor, current_state, feature)
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

            prefix_go = tf.concat([tf.ones([prefix_batch_tensor, 1], dtype=tf.int32) * GO_ID, self.prefix],
                                  axis=1)

            prefix_embedding = tf.nn.embedding_lookup(embedding, prefix_go)
            _, current_state = tf.nn.dynamic_rnn(cell, prefix_embedding, dtype=tf.float32)
            outputs_infer = []

            input_tensor = tf.nn.embedding_lookup(embedding, self.prefix[:, -1])
            for time_step in range(max_length):
                with tf.variable_scope('cell_model', reuse=True) as cell_scope:
                    feature = self.fixed_feature
                    output, current_state = cell_model(cell, input_tensor, current_state, feature)
                    output = output * (1 - tf.one_hot(2, vocab_size))
                    output_voc = tf.argmax(output, axis=1)
                    input_tensor = tf.nn.embedding_lookup(embedding, output_voc)
                    outputs_infer.append(tf.reshape(output_voc, [batch_tensor, 1]))
            self.result_infer = tf.concat(outputs_infer, axis=1, name="output_infer")

            tf.summary.scalar('perplexity_train', self.perplexity)
            tf.summary.scalar('loss_train', self.loss)
            self.summaries = tf.summary.merge_all()



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



    def update(self, sess, beta, reader):
        batch = reader.get_batch(self.batch_size)
        if batch == None:
            reader.reset()
            batch = reader.get_batch(self.batch_size)
        # build feed dict
        feed_truth, feed_weight, feed_feature = reader.process_batch(batch, self.max_length)
        feed_dict = {}

        feed_dict[self.ground_truth] = np.array(feed_truth, dtype=np.int32)
        feed_dict[self.target_weight] = np.array(feed_weight, dtype=np.float32)
        feed_dict[self.feature] = np.array(feed_feature, dtype=np.float32)
        feed_dict[self.beta] = beta

        feed_output = [self.result, self.loss, self.perplexity, self.summaries, self.train]
        result, loss, perplexity, summaries, _ = sess.run(feed_output, feed_dict=feed_dict)

        # reader.output(result, batch)

        print 'perplexity = ' + str(perplexity)
        return summaries
    
    def inference(self, sess, beta, prefix_size, fixed_feature, reader, file):

        batch = reader.get_batch(self.batch_size, prefix_size)
        feed_prefix, feed_weight, feed_feature = reader.process_batch(batch, prefix_size)

        feed_dict = {}
        feed_dict[self.prefix] = np.array(feed_prefix, dtype=np.int32)
        feed_dict[self.target_weight] = np.array(feed_weight, dtype=np.float32)
        feed_dict[self.fixed_feature] = np.array(fixed_feature, dtype=np.float32)
        feed_dict[self.beta] = beta

        print 'prefix:', feed_dict[self.prefix].shape
        print 'weight:', feed_dict[self.target_weight].shape

        result_infer = sess.run(self.result_infer, feed_dict=feed_dict)
        result_prefix = [tmp[0] for tmp in batch]
        # print result_prefix, result_infer
        reader.output([np.concatenate([prefix, infer]) for prefix, infer in zip(result_prefix, result_infer)], file=file)

        return result_infer

if __name__ == "__main__":
    model = Affect_LM_Model(30000)
    model.all_params()