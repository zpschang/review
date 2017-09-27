from reader import Reader
from model import Affect_LM_Model
import tensorflow as tf
import json
word_filename = 'dataset/words.txt'

reader = Reader('dataset/douban/comments/1578714.json', word_filename)
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


string = open('dataset/douban/movie_list.json', 'r').read()

# TODO : training in different movies
"""
for obj in json.loads(string):
    movie_id = obj['id']
    movie_url = obj['url']
    filename = 'dataset/douban/comments/' + movie_id + '.json'
"""

print '=========start training========='

try:
    for _ in range(100):
        model.update(sess, 1.0, reader)
    saver.save(sess, 'model/affect-lm.ckpt')

except KeyboardInterrupt:
    saver.save(sess, 'model/affect-lm.ckpt')