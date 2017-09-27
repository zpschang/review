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

class Reader:
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
        print self.catagory

    def get_batch(self, batch_size, prefix_size=None):
        if self.index + batch_size > len(self.object):
            return None
        object = self.object[self.index : self.index + batch_size]
        batch = []

        # TODO: filter out over-length comment

        for comment in object:
            str_comment = comment['comment_str']
            point = comment['point']
            words_raw = jieba.cut(str_comment)
            words = []
            for word in words_raw:
                if word not in {' ', '\n'}:
                    words.append(word)

            int_comment = [self.word_to_index[word] if word in self.word_to_index else UNK_ID for word in words]
            int_comment = int_comment + [EOS_ID]
            if prefix_size != None:
                int_comment = int_comment[:prefix_size]
            catagory = self.get_catagory(int_comment)
            batch.append((int_comment, catagory, point))
        self.index += batch_size
        return batch

    def get_catagory(self, comment):
        # print self.word_to_catagory
        catagory_batch = []
        for word in comment:
            str_word = self.words[word]

            catagories = self.word_to_catagory[str_word] if str_word in self.word_to_catagory else []
            one_hot = [0] * len(self.catagory)
            for index in catagories:
                # print self.catagory[index], str_word
                one_hot[index] = 1
            catagory_batch.append(one_hot)
        for index in reversed(range(len(catagory_batch))):
            lower_bound = max([0, index - sentiment_length + 1])
            related = np.array(catagory_batch[lower_bound: index + 1])

            catagory_batch[index] = [float(max(related[:, i])) for i in range(len(self.catagory))]
        return catagory_batch

    def output(self, result, batch=None):
        length = len(result) if result != None else len(batch)
        for index in range(length):
            def output(comment):
                for word in comment:
                    print self.words[word],
                    if word == EOS_ID:
                        break
                print '\n',
            if batch != None:
                print 'truth:'
                output(batch[index][0])
            if result != None:
                print 'result:'
                output(result[index])

    def set_review(self, review_filename):
        string = open(review_filename, 'rb').read()
        self.object = json.loads(string)
        self.index = 0
        self.catagory, self.word_to_catagory = load_catagory()

    def reset(self):
        self.index = 0
