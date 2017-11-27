import tensorflow as tf
import os
from reader import SimpleReader
from model_aspect import Aspect_LM_Model
from tensorflow.python.client import device_lib
import sys

def main():
    model_dir = 'model_aspect_new'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    print device_lib.list_local_devices()

    word_filename = 'dataset/words.txt'
    review_filename = 'dataset/aspect_test.txt'
    reader = SimpleReader(word_filename, review_filename)

    print len(reader.reviews)
    batch_size = len(reader.reviews)

    print '=========initialize model========='
    model = Aspect_LM_Model(vocab_size=len(reader.words), batch_size=batch_size)

    print '=========load environment========='

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    try:
        loader = tf.train.import_meta_graph(model_dir + '/aspect-lm.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint(model_dir + '/'))
        print 'load finished'
    except:
        print 'load failed'
        return

    model.inference(sess, 3, reader)

if __name__ == '__main__':
    main()
