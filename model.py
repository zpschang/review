import tensorflow as tf
from tensorflow.contrib import seq2seq

class Affect_LM_Model():
    def __init__(self,
                vocab_size,
                embedding_size,
                lstm_size,
                num_layer,
                max_length,
                feature_size,
                max_gradient_norm,
                batch_size,
                learning_rate,
                beam_width,
                embed=None):
        self.batch_size = batch_size
        self.max_length = max_length

        with tf.variable_scope("Affect-LM") as scope:
            self.ground_truth = tf.placeholder(tf.int32, [None, max_length], "ground_truth")
            self.target_weight = tf.placeholder(tf.float32, [None, max_length], "target_weight")
            self.feature = tf.placeholder(tf.float32, [None, max_length, feature_size], "feature")

            embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)




    def update(self):
        pass
    
    def inference(self):
        pass