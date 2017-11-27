import json
import jieba
import re
from analyze import load_category
import numpy as np
GO_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3
sentiment_length = 3

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
        self.category, self.word_to_category = load_category()
        print self.category

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
            category = self.get_category(int_comment)
            batch.append((int_comment, category, point))
        return batch

    def get_category(self, comment):
        # print self.word_to_category
        category_batch = []
        for word in comment:
            str_word = self.words[word]

            catagories = self.word_to_category[str_word] if str_word in self.word_to_category else []
            one_hot = [0] * len(self.category)
            for index in catagories:
                # print self.category[index], str_word
                one_hot[index] = 1
            category_batch.append(one_hot)
        result = [None for _ in range(len(comment))]
        for index in reversed(range(len(category_batch))):
            lower_bound = max([0, index - sentiment_length])
            upper_bound = min(index + sentiment_length, len(category_batch))
            """
            if index > 0:
                related = np.array(category_batch[lower_bound: index])
                category_batch[index] = np.sum(related, axis=0).astype(float)
            else:
                category_batch[index] = np.zeros([len(self.category)], dtype=np.float)
            """
            related = np.array(category_batch[lower_bound: upper_bound])
            result[index] = np.sum(related, axis=0).astype(float)
            # category_batch[index] = [float(max(related[:, i])) for i in range(len(self.category))]
        return result

    def output(self, result, batch=None, file=None):
        length = len(result) if result is not None else len(batch)
        string = ''
        for index in range(length):
            if batch is not None:
                string = string + 'truth:\n'
                for word in batch[index][0]:
                    string = string + self.words[word] + ' '
                    if word == EOS_ID:
                        break
                string += '\n'
            if result is not None:
                string = string + 'result:\n'
                for word in result[index]:
                    string = string + self.words[word] + ' '
                    if word == EOS_ID:
                        break
                string += '\n'
        if file is None:
            print string.encode('utf-8')
        else:
            file.write(string.encode('utf-8'))

    def reset(self):
        self.index[self.review_filename] = 0

    def set_review(self, review_filename):
        self.review_filename = review_filename
        if review_filename not in self.object:
            file = open(review_filename, 'rb')
            string = file.read()
            self.object[review_filename] = json.loads(string)
            file.close()

    def process_batch(self, batch, max_length):
        feed_truth = []
        feed_weight = []
        feed_feature = []
        feature_size = len(self.category)
        for comment, category, point in batch:
            pad_num = max_length - len(comment)
            weight = [1.0] * len(comment) + [0.0] * pad_num
            for index in range(len(comment)):
                if comment[index] == UNK_ID:
                    weight[index] = 0

            category = category + [[0.0] * feature_size for _ in range(pad_num)]

            # add point
            point_vector = [1 if index == point-1 else 0 for index in range(5)]
            for index in range(len(category)):
                if type(category[index]) is np.ndarray:
                    category[index] = category[index].tolist() + point_vector
                else:
                    category[index] = category[index] + point_vector
            comment = comment + [PAD_ID] * pad_num
            if len(comment) > max_length:
                weight = weight[:max_length]
                category = category[:max_length]
                comment = comment[:max_length]
            feed_truth.append(comment)
            feed_weight.append(weight)
            feed_feature.append(category)
        return feed_truth, feed_weight, feed_feature

class Simple_reader():
    def __init__(self, word_filename, review_filename):
        self.words = []
        self.word_to_index = {}
        for word in open(word_filename, 'rb').readlines():
            word = word[:-1]
            self.word_to_index[word] = len(self.words)
            self.words.append(word)

        self.object = {}
        self.index = {}
        self.review_filename = review_filename
        string = open(review_filename, 'rb').readlines()
        self.reviews = []
        self.rating = []
        self.aspect = []

        print 'start processing data'
        for index in range(0, len(string), 2):
            line_review = string[index]
            line_score = string[index + 1]
            words = re.split(' ', line_review)

            if words[-1][-1] == '\n':
                words[-1] = words[-1][:-1]

            review_index = [self.word_to_index[word] if word in self.word_to_index else UNK_ID for word in words]
            review_index += [EOS_ID]
            pair = re.findall('-?[0-9]', line_score)

            aspect = eval(pair[0])
            rating = eval(pair[1])
            self.reviews.append(review_index)
            self.aspect.append(aspect)
            self.rating.append(rating)

        print 'finish processing data'
        self.index = 0
    def get_batch(self, batch_size, prefix_size=None):
        batch = []
        for _ in range(batch_size):
            if self.index == len(self.reviews):
                self.index = 0
            if prefix_size is None:
                reviews = self.reviews[self.index]
            else:
                reviews = self.reviews[self.index][:prefix_size]
            rating = self.rating[self.index]
            aspect = self.aspect[self.index]
            batch.append((reviews, rating, aspect))

            self.index += 1

        return batch

    def output(self, result, batch=None, file=None):
        length = len(result) if result is not None else len(batch)
        string = ''
        for index in range(length):
            if batch is not None:
                string = string + 'truth:\n'
                for word in batch[index][0]:
                    string = string + self.words[word] + ' '
                    if word == EOS_ID:
                        break
                string += '\n'
            if result is not None:
                string = string + 'result:\n'
                for word in result[index]:
                    string = string + self.words[word] + ' '
                    if word == EOS_ID:
                        break
                string += '\n'
        if file is None:
            print string
        else:
            file.write(string)

    def process_batch(self, batch, max_length):
        feed_truth = []
        feed_weight = []
        feed_rating = []
        feed_aspect = []

        for comment, rating, aspect in batch:
            feed_rating.append(rating)
            feed_aspect.append(aspect)
            pad_num = max_length - len(comment)
            weight = [1.0] * len(comment) + [0.0] * pad_num
            for index in range(len(comment)):
                if comment[index] == UNK_ID:
                    weight[index] = 0

            comment = comment + [PAD_ID] * pad_num
            if len(comment) > max_length:
                weight = weight[:max_length]
                comment = comment[:max_length]
            feed_truth.append(comment)
            feed_weight.append(weight)

        return feed_truth, feed_weight, feed_rating, feed_aspect