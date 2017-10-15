from reader import Reader
from model import Affect_LM_Model
import tensorflow as tf
import json
word_filename = 'dataset/words.txt'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

reader = Reader(word_filename)
batch_size = 40

print '=========initialize model========='
model = Affect_LM_Model(vocab_size=len(reader.words), feature_size=len(reader.catagory))

print '=========load environment========='
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
try:
    loader = tf.train.import_meta_graph('model/affect-lm.ckpt.meta')
    loader.restore(sess, tf.train.latest_checkpoint('model/'))
    print 'load finished'
except:
    sess.run(tf.global_variables_initializer())
    print 'load failed'

# TODO : training in different movies

print '=========start training========='

try:
    for iteration in range(1000):
        string = open('dataset/douban/movie_list.json', 'r').read()
        for obj in json.loads(string):
            movie_id = obj['id']
            filename = 'dataset/douban/comments/' + movie_id + '.json'
            try:
                reader.set_review(filename)
                print 'iteration:', iteration, 'movie:', movie_id
                model.update(sess, 1.0, reader)
            except IOError:
                pass
        saver.save(sess, 'model/affect-lm.ckpt')

except KeyboardInterrupt:
    saver.save(sess, 'model/affect-lm.ckpt')