# coding=utf-8

import json
import re
import string

# load category dictionary for the use of Affect-LM
def load_category():
    f = open('dataset/sc_liwc_sentiment.dic', 'r')
    category = []
    category_to_index = {}
    word_to_category = {}
    index_to_words = []
    f.readline()
    num = 0
    while True:
        line = f.readline().decode('utf-8')
        if line[0] == '%':
            break
        print line
        str_num, str_cata = re.split('[ \t]', line)
        category.append(str_cata[:-2])
        index_to_words.append([])
        category_to_index[int(str_num)] = num
        num += 1
    while True:
        line = f.readline().decode('utf-8')
        if line == '':
            break
        tmp = re.split('[ \t]', line)
        word = tmp[0]
        if word[-1] == '*':
            word = word[:-1]

        word_to_category[word] = [string.atoi(str_num) for str_num in tmp[1:]]

    # word_to_category = {key: category_to_index[word_to_category[key]] for key in word_to_category}
    for word in word_to_category:
        index = []
        for item in word_to_category[word]:
            if item in category_to_index:
                index.append(category_to_index[item])
                index_to_words[category_to_index[item]].append(word)
        # index = [category_to_index[item] for item in word_to_category[word]]
        word_to_category[word] = index
    for index in range(len(category)):
        print category[index], ':'
        for word in index_to_words[index]:
            print word.encode('utf-8'),
        print '\n',
    return category, word_to_category


# loading yelp dataset. temporarily unused
def load_dataset():
    # f = open('dataset/reviews_Movies_and_TV_5.json', 'r')
    f = open('dataset/review.json', 'r')
    stat = {}

    #for line in f.readlines():
    for i in range(100):
        line = f.readline()
        d = json.loads(line)
        print d
        if d['business_id'] not in stat:
            stat[d['business_id']] = {'num': 1}
        else:
            stat[d['business_id']]['num'] += 1
    result = sorted(stat.items(), key=lambda x : x[1]['num'])
    print result

if __name__ == '__main__':
    category, word_to_category = load_category()

    print word_to_category['生气'.decode('utf-8')]