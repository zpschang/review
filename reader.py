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

# read douban movie review and generate mini-batch
class Reader:
    def __init__(self, word_filename):
        self.words = []
        self.word_to_index = {}
        for word in open(word_filename, 'rb').readlines():
            word = word.decode('utf-8')[:-1]
            self.word_to_index[word] = len(self.words)
            self.words.append(word)
        self.object = {}
        self.index = {}
        self.catagory, self.word_to_catagory = load_catagory()
        print self.catagory

    def filter(self, comment): # whether this comment is selected
        # TODO: filter
        return True

    def get_batch(self, batch_size, prefix_size=None):
        batch = []
        object = []
        if self.review_filename not in self.index:
            self.index[self.review_filename] = 0
        object_review = self.object[self.review_filename]
        while len(object) < batch_size:
            str_comment = object_review[self.index[self.review_filename]]
            if self.filter(str_comment):
                object.append(object_review[self.index[self.review_filename]])
            self.index[self.review_filename] += 1
            self.index[self.review_filename] %= len(object_review)

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
        length = len(result) if result is not None else len(batch)
        for index in range(length):
            def output(comment):
                for word in comment:
                    print self.words[word].encode('utf-8'),
                    if word == EOS_ID:
                        break
                print '\n',
            if batch is not None:
                print 'truth:'
                output(batch[index][0])
            if result is not None:
                print 'result:'
                output(result[index])

    def reset(self):
        self.index[self.review_filename] = 0

    def set_review(self, review_filename):
        self.review_filename = review_filename
        string = open(review_filename, 'rb').read()
        if review_filename not in self.object:
            self.object[review_filename] = json.loads(string)