import tensorflow as tf
import os
from reader import Reader
from model import Affect_LM_Model
from tensorflow.python.client import device_lib
import sys
from scipy.signal import convolve2d


model_dir = 'model_affect_context'

def main():
    strength = eval(sys.argv[1])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print device_lib.list_local_devices()

    word_filename = 'dataset/words.txt'
    reader = Reader(word_filename)
    batch_size = 40

    print '=========initialize model========='
    model = Affect_LM_Model(vocab_size=len(reader.words), feature_size=len(reader.category)+5)

    print '=========load environment========='

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    try:
        loader = tf.train.import_meta_graph(model_dir + '/affect-lm.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint(model_dir + '/'))
        print 'load finished'
    except:
        print 'load failed'
        return

    review_filename = 'dataset/evaluate/test.json'
    reader.set_review(review_filename)
    batch_size = len(reader.object[review_filename])
    model.batch_size = batch_size
    result_infer = model.inference(sess, strength, prefix_size=5, fixed_feature=[[1,1,0,0,0,0,0,0,0,0,1]] * batch_size, reader=reader)

if __name__ == '__main__':
    main()
