import json
import jieba
from analyze import load_catagory
import numpy as np
EOS_ID = 1
UNK_ID = 2
sentiment_length = 7

class reader_amazon:
    def __init__(self, review_filename):
        pass
    def get_batch(self, batch_size):
        pass

class reader:
    def __init__(self, review_filename, word_filename):
        self.words = []
        self.word_to_index = {}
        for word in open(word_filename, 'rb').readlines():
            word = word.decode('utf-8')[:-1]
            self.word_to_index[word] = len(self.words)
            self.words.append(word)
        string = open(review_filename, 'rb').read()
        self.object = json.loads(string)
        self.index = 0
        self.catagory, self.word_to_catagory = load_catagory()

    def get_batch(self, batch_size, prefix_size=None):
        if self.index + batch_size > len(self.object):
            return None
        object = self.object[self.index, self.index + batch_size]
        batch = []
        for comment in batch:
            str_comment = comment['comment_str']
            point = comment['point']
            int_comment = [self.word_to_index[word] if word in self.word_to_index else UNK_ID for word in jieba.cut(str_comment)]
            int_comment = int_comment + [EOS_ID]
            if prefix_size != None:
                int_comment = int_comment[:prefix_size]
            catagory = self.get_catagory(int_comment)
            batch.append(tuple(int_comment, catagory, point))
        self.index += batch_size
        return batch

    def get_catagory(self, comment):
        catagory_batch = []
        for word in comment:
            str_word = self.words[word]
            catagories = self.word_to_catagory[str_word] if str_word in self.word_to_catagory else []
            one_hot = [0] * len(self.catagory)
            for index in catagories:
                one_hot[index] = 1
            catagory_batch.append(one_hot)
        for index in reversed(range(len(catagory_batch))):
            related = np.array(catagory_batch[index - sentiment_length + 1: index + 1])
            catagory_batch[index] = [any(related[:, i]) for i in range(len(self.catagory))]
        return catagory_batch

    def output(self, result, batch):
        for (comment_truth, _, _), comment_generate in zip(batch, result):
            def output(comment):
                for word in comment:
                    print self.words[word],
                    if word == EOS_ID:
                        break
            print 'truth:'
            output(comment_truth)
            print 'generate:'
            output(comment_generate)

