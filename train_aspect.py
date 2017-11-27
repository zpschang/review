from reader import Reader, SimpleReader
from model_aspect import Aspect_LM_Model
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.python.client import device_lib
from tensorflow.python import debug
import sys
import time

model_dir = sys.argv[1]
assert os.path.isdir(model_dir)

def main():
    # tensorboard
    writer = tf.summary.FileWriter('/tmp/tensorboard_zps/aspect-lm')

    batch_size = 20
    try:
        file_plot = open('plot.obj', 'rb')
        data_to_plot = pickle.load(file_plot)
        file_plot.close()
    except:
        data_to_plot = {}
    if model_dir not in data_to_plot:
        data_to_plot[model_dir] = []

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    print device_lib.list_local_devices()

    word_filename = 'dataset/words.txt'
    review_filename = 'dataset/aspect_shuffle.txt'

    reader = SimpleReader(word_filename, review_filename)

    test_filename = 'dataset/aspect_test.txt'
    reader_test = SimpleReader(word_filename, test_filename)

    print '=========initialize model========='
    model = Aspect_LM_Model(vocab_size=len(reader.words), batch_size=batch_size)

    print '=========load environment========='
    sess = tf.Session()
    # sess = debug.LocalCLIDebugWrapperSession(sess)

    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    try:
        loader = tf.train.import_meta_graph(model_dir + '/aspect-lm.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint(model_dir + '/'))
        print 'load finished'
    except:
        sess.run(tf.global_variables_initializer())
        data_to_plot[model_dir] = []
        print 'load failed'

    writer.add_graph(sess.graph)

    print '=========start training========='
    print data_to_plot

    try:
        for iteration in range(10000000):
            start_time = time.time()
            summaries = model.update(sess, reader)
            global_step = sess.run(model.global_step)
            print 'global_step:', global_step
            writer.add_summary(summaries, global_step=global_step)
            print 'progress:', iteration * model.batch_size, '/', 620840
            # data_to_plot[model_dir].append(perplexity)
            print 'batch training time:', time.time() - start_time, 's'
            if iteration % 10000 == 0:
                saver.save(sess, model_dir + '/aspect-lm.ckpt')
                plot(data_to_plot)
                file = open('result/result_' + str(iteration) + '.txt', 'w')
                model.batch_size = len(reader_test.reviews)
                model.inference(sess, prefix_size=3, reader=reader_test, file=file)
                file.close()
                model.batch_size = batch_size

    except KeyboardInterrupt:
        saver.save(sess, model_dir + '/aspect-lm.ckpt')
        file_plot = open('plot.obj', 'wb')
        pickle.dump(data_to_plot, file_plot)
        file_plot.close()
        plot(data_to_plot)
    saver.save(sess, model_dir + '/aspect-lm.ckpt')
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