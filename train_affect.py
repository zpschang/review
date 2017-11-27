from reader import Reader
from model_affect import Affect_LM_Model
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.python.client import device_lib
import sys
import time

model_dir = sys.argv[1]
assert os.path.isdir(model_dir)

affect_vec = [[1,1,0,0,0,0],
              [1,0,1,0,0,0],
              [1,0,1,1,0,0],
              [1,0,1,0,1,0],
              [1,0,1,0,0,1]]
test_pair = [(0, 5), (0, 4), (0, 3), (1, 3), (1, 2), (1, 1)]

def main():
    # tensorboard
    writer = tf.summary.FileWriter('/tmp/tensorboard_zps/affect-lm')

    try:
        file_plot = open('plot.obj', 'rb')
        data_to_plot = pickle.load(file_plot)
        file_plot.close()
    except:
        data_to_plot = {}
    if model_dir not in data_to_plot:
        data_to_plot[model_dir] = []

    batch_num = 0

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    print device_lib.list_local_devices()

    word_filename = 'dataset/words.txt'
    reader = Reader(word_filename)
    batch_size = 20

    print '=========initialize model========='
    model = Affect_LM_Model(vocab_size=len(reader.words), feature_size=len(reader.category)+5, is_pure_LM=True)

    print '=========load environment========='
    sess = tf.Session()

    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    try:
        loader = tf.train.import_meta_graph(model_dir + '/affect-lm.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint(model_dir + '/'))
        print 'load finished'
    except:
        sess.run(tf.global_variables_initializer())
        data_to_plot[model_dir] = []
        print 'load failed'

    print '=========start training========='
    print data_to_plot

    try:
        for iteration in range(1000):
            string = open('dataset/douban/movie_list.json', 'r').read()
            num = 0
            review_filename = 'dataset/evaluate/test.json'
            reader.set_review(review_filename)
            for affect_index, rating in test_pair:
                # infer
                file = open('result/result_%d_%d.txt' % (iteration, num), 'w')
                file.write('affect: %d\nrating: %d\n' % (affect_index, rating))

                model.batch_size = len(reader.object[review_filename])
                rating_vec = [0] * 5
                rating_vec[rating - 1] = 1
                feature_vec = affect_vec[affect_index] + rating_vec
                result_infer = model.inference(sess, 1.0, prefix_size=5,
                                               fixed_feature=[feature_vec] * model.batch_size,
                                               reader=reader, file=file)
                file.close()
                reader.reset()
                num += 1
            model.batch_size = batch_size

            trained = set()
            for obj in json.loads(string):
                movie_id = obj['id']
                if movie_id in trained:
                    continue
                trained.add(movie_id)
                filename = 'dataset/douban/comments/' + movie_id + '.json'
                try:
                    reader.set_review(filename)
                    print 'round:', iteration, 'batch_num:', batch_num
                    if len(reader.object[filename]) > 0:
                        start_time = time.time()
                        summaries = model.update(sess, 1.0, reader)
                        # data_to_plot[model_dir].append(perplexity)
                        global_step = sess.run(model.global_step)
                        print 'global_step:', global_step
                        writer.add_summary(summaries, global_step=global_step)

                        print 'batch training time:', time.time() - start_time, 's'
                        batch_num += 1
                except IOError:
                    pass
            file_plot = open('plot.obj', 'wb')
            pickle.dump(data_to_plot, file_plot)
            file_plot.close()
            plot(data_to_plot)

    except KeyboardInterrupt:
        saver.save(sess, model_dir + '/affect-lm.ckpt')
        file_plot = open('plot.obj', 'wb')
        pickle.dump(data_to_plot, file_plot)
        file_plot.close()
        plot(data_to_plot)

def plot(data_to_plot):
    plt.title('Perplexity Graph')
    plt.xlabel('iterations')
    plt.ylabel('perplexity')
    plt.yscale('log')
    for label, list_perplexity in data_to_plot.items():
        plt.plot(list_perplexity, label=label, alpha=0.5)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6.7)
    fig.savefig('graph/perplexity.png', dpi=100)
    plt.close()

if __name__ == '__main__':
    main()