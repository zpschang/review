import tensorflow as tf
from tensorflow.python.client import device_lib
from model_affect import Affect_LM_Model
from model_aspect import Aspect_LM_Model
from model_hd import HD_model
from model_hdn import HDN_model
from reader import SimpleReader
from parameter import hyper
import os
import time

word_filename = 'dataset/words.txt'
review_filename = 'dataset/aspect_shuffle.txt'
test_filename = 'dataset/aspect_test.txt'

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # print device_lib.list_local_devices()

    # building reader
    reader = SimpleReader(word_filename, review_filename, is_train=True)
    reader_test = SimpleReader(word_filename, test_filename, is_train=False)
    hyper.vocab_size = len(reader.words)

    # building model
    model_name = ['affect', 'aspect', 'hd', 'hdn', 'affect']
    model_class = [Affect_LM_Model, Aspect_LM_Model, HD_model, HDN_model]
    name_to_class = {name: class_ for name, class_ in zip(model_name, model_class)}
    print 'trainable models:'
    print ' '.join(model_name)
    input_str = raw_input('model:')
    if input_str not in model_name:
        print 'wrong model name'
        return
    print 'building model'
    model = name_to_class[input_str]()
    print 'build finished'

    # load environment
    model_dir = 'model/' + input_str
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    try:
        loader = tf.train.import_meta_graph(model_dir + '/model.ckpt.meta')
        loader.restore(sess, tf.train.latest_checkpoint(model_dir + '/'))
        print 'load finished'
    except:
        sess.run(tf.global_variables_initializer())
        print 'load failed'


    # start training
    result_dir = 'result/' + input_str
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    summary_dir = '/tmp/tensorboard_zps/' + input_str
    if not os.path.isdir(summary_dir):
        os.mkdir(summary_dir)
    # tensorboard
    writer = tf.summary.FileWriter(summary_dir)
    writer.add_graph(sess.graph)

    try:
        for iteration in range(10000000):
            start_time = time.time()
            summaries = model.update(sess, reader)
            global_step = sess.run(model.global_step)
            print 'global_step:', global_step
            writer.add_summary(summaries, global_step=global_step)
            print 'progress:', reader.index, '/', len(reader.reviews)
            # data_to_plot[model_dir].append(perplexity)
            print 'batch training time:', time.time() - start_time, 's'
            if iteration % 10000 == 0:
                saver.save(sess, model_dir + '/model.ckpt')
                file = open(result_dir + '/result_' + str(iteration) + '.txt', 'w')
                model.batch_size = len(reader_test.reviews)
                model.inference(sess, reader=reader_test, file=file)
                file.close()
                model.batch_size = hyper.batch_size

    except KeyboardInterrupt:
        saver.save(sess, model_dir + '/model.ckpt')
        file_plot = open('plot.obj', 'wb')
        file_plot.close()
    saver.save(sess, model_dir + '/model.ckpt')
    file_plot = open('plot.obj', 'wb')
    file_plot.close()

if __name__ == '__main__':
    main()