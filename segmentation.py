import jieba
import jieba.posseg
import json
import matplotlib.pyplot as plt
import sys
import re
# word segmentation & vocabulary generation
# run this program before training to generate vocabulary

# a, ag, an, ad: adjective(affect)
# c, p: conjunction, preposition(relation)
# n, nr, ns, nt, nz, nrfg: noun(aspect)
# r: pronoun(???)
# u* : auxiliary

# eng, m

def get_all_words():
    words = {}
    string = open('dataset/douban/movie_list.json', 'rb').read()
    num = 0
    objects = json.loads(string)
    for obj in objects:
        movie_id = obj['id']
        movie_url = obj['url']
        print num, '/', len(objects)
        print 'word len:', len(words)
        filename = 'dataset/douban/comments/' + movie_id + '.json'
        try:
            string_comments = open(filename, 'rb').read()
        except:
            num += 1
            continue
        object_comments = json.loads(string_comments)
        for comment in object_comments:
            string_comment = comment[u'comment_str']
            for word in jieba.cut(string_comment):
                if word not in words:
                    words[word] = 1
                else:
                    words[word] += 1
        num += 1

    return words

def plot_dataset():
    words = get_all_words()
    now_sum = 0
    total_sum = 0
    percentage = []
    for key, value in words.items():
        if key == ' ':
            continue
        total_sum += value
    for key, value in sorted(words.items(), key=lambda x: x[1], reverse=True):
        if key == ' ':
            continue
        now_sum += value
        percentage.append(1.0 * now_sum / total_sum)
    plt.title('percentage - words')
    plt.ylim(ymin=0.9, ymax=1)
    plt.plot(percentage)
    plt.savefig('frequency.png', dpi=100)

def plot_length():
    string = open('dataset/douban/movie_list.json', 'r').read()
    num = 0
    objects = json.loads(string)
    list_length = []
    for obj in objects:
        movie_id = obj['id']
        movie_url = obj['url']
        print num, '/', len(objects)
        filename = 'dataset/douban/comments/' + movie_id + '.json'
        try:
            string_comments = open(filename, 'r').read()
        except:
            break
        object_comments = json.loads(string_comments)

        for comment in object_comments:
            string_comment = comment[u'comment_str']
            string_comment = re.sub('[ \n\r]', '', string_comment)
            word_num = 0
            for _ in jieba.cut(string_comment):
                word_num += 1
            list_length.append(word_num)
        num += 1

    plt.title('length histogram', fontsize=20)
    plt.xlabel('total words', fontsize=15)
    plt.ylabel('review number', fontsize=15)
    plt.axis(xmin=0, xmax=150)
    plt.hist(list_length, bins=50, range=(0, 150), facecolor='yellowgreen', alpha=0.75)
    fig = plt.gcf()
    fig.set_size_inches(12, 6.7)
    fig.savefig('length_histogram.png', dpi=100)

def generate_word():
    words = get_all_words()
    num = 0
    total_sum = 0
    validate_sum = 0
    file_word = open('dataset/words.txt', 'wb')
    def write(word, type):
        file_word.write('%s %s\n' % (word, type))
    write('<go>', 'special')
    write('<eos>', 'special')
    write('<unk>', 'special')
    write('<pad>', 'special')
    for key, value in sorted(words.items(), key=lambda x: x[1], reverse=True):
        if key in {' ', '\n'}:
            continue
        total_sum += value
        if value < 15:
            continue
        validate_sum += value
        num += 1
        if num < 500:
            print key.encode('utf-8'),
        result = list(jieba.posseg.cut(key))
        write(key.encode('utf-8'), result[0].flag.encode('utf-8'))

    print num
    print 1.0 * validate_sum / total_sum

if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        print 'plot word frequency'
        plot_dataset()
    elif sys.argv[1] == 'word':
        print 'word generation'
        generate_word()
    elif sys.argv[1] == 'length':
        print 'plot length'
        plot_length()
    else:
        words, tags = get_all_words()
        print tags