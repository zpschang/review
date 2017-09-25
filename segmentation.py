import jieba
import json

# word segmentation
# vocabulary generation
# filter out low quality comments

def get_all_words():
    words = {}
    string = open('dataset/douban/movie_list.json', 'r').read()

    for obj in json.loads(string):
        movie_id = obj['id']
        movie_url = obj['url']
        print movie_id
        print 'word len:', len(words)
        filename = 'dataset/douban/comments/' + movie_id + '.json'
        try:
            string_comments = open(filename, 'r').read()
        except:
            break
        object_comments = json.loads(string_comments)
        for comment in object_comments:
            string_comment = comment[u'comment_str']
            for word in jieba.cut(string_comment):
                if word not in words:
                    words[word] = 1
                else:
                    words[word] += 1
    return words


if __name__ == '__main__':
    words = get_all_words()
    num = 0
    total_sum = 0
    validate_sum = 0
    file_word = open('dataset/words.txt', 'wb')
    file_word.write('<go>\n<eos>\n<unk>\n<pad>\n')
    for key, value in sorted(words.items(), key=lambda x: x[1]):
        if key == ' ':
            break
        total_sum += value
        if value < 5:
            continue
        validate_sum += value
        num += 1
        print '(' + str([key]) + ': ' + str(value) + '),',
        file_word.write(key.encode('utf-8') + '\n')



    print num
    print 1.0 * validate_sum / total_sum

