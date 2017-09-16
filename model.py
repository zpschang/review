import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops

GO_ID = 0

class Affect_LM_Model():
    def __init__(self,
                vocab_size,
                embedding_size = 128,
                lstm_size = 200,
                num_layer = 4,
                max_length = 40,
                feature_size = 6,
                max_gradient_norm = 2,
                batch_size = 20,
                learning_rate = 0.25,
                beam_width = 5,

                embed=None):
        self.batch_size = batch_size
        self.max_length = max_length

        with tf.variable_scope("Affect-LM") as scope:
            self.ground_truth = tf.placeholder(tf.int32, [None, max_length], "ground_truth")
            self.target_weight = tf.placeholder(tf.float32, [None, max_length], "target_weight")
            self.feature = tf.placeholder(tf.float32, [None, max_length, feature_size], "feature") # for training
            self.fixed_feature = tf.placeholder(tf.float32, [None, feature_size], "fixed_feature") # for inferring
            self.prefix = tf.placeholder(tf.int32, [None, None], "prefix")
            self.beta = tf.placeholder(tf.float32, [], "beta")

            batch_tensor = tf.shape(self.ground_truth)[0]

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
                feature = feature * self.beta
                feature_output = tf.nn.relu(core.dense(feature, 100))
                feature_output = tf.nn.relu(core.dense(feature, 200))
                output = tf.concat([cell_output, feature_output], axis=1)
                output = core.dense(output, vocab_size)
                return output, current_state

            outputs_train = []
            for time_step in range(max_length):
                with tf.variable_scope('cell_model') as cell_scope:
                    if time_step > 0:
                        cell_scope.reuse_variables()
                    print 'time step:', time_step
                    input_tensor = input_embedding[:, time_step, :]
                    feature = self.feature[:, time_step, :]
                    output, current_state = cell_model(cell, input_tensor, current_state, feature)
                    # TODO: copy output if <eos> occurs
                    outputs_train.append(tf.reshape(output, [batch_tensor, 1, vocab_size]))

            logits = tf.concat(outputs_train, axis=1, name="final_output")
            self.output = tf.argmax(logits, 2, name="output")
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=logits)
            self.loss = tf.reduce_sum(cross_entropy * self.target_weight) / tf.cast(batch_tensor, tf.float32)
            self.perplexity = tf.exp(tf.reduce_sum(cross_entropy * self.target_weight) / tf.reduce_sum(self.target_weight))

            # TODO: not elegant. can be replaced with tf.while_loop
            prefix_embedding = tf.nn.embedding_lookup(embedding, self.prefix)
            _, current_state = tf.nn.dynamic_rnn(cell, prefix_embedding, dtype=tf.float32)
            outputs_infer = []

            input_tensor = tf.nn.embedding_lookup(embedding, self.prefix[:, -1])
            for time_step in range(max_length):
                with tf.variable_scope('cell_model', reuse=True) as cell_scope:
                    print 'time step:', time_step
                    feature = self.fixed_feature
                    output, current_state = cell_model(cell, input_tensor, current_state, feature)

                    output_voc = tf.argmax(output, axis=1)
                    input_tensor = tf.nn.embedding_lookup(embedding, output_voc)
                    outputs_infer.append(tf.reshape(output_voc, [batch_tensor, 1]))
            self.output_infer = tf.concat(outputs_infer, axis=1, name="output_infer")




    def all_params(self):
        with tf.variable_scope('Affect-LM') as scope:
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

    def update(self):
        pass
    
    def inference(self):
        pass

if __name__ == "__main__":
    model = Affect_LM_Model(30000)
    model.all_params()