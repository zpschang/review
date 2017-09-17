import json
import re
import string

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

def load_catagory():
    f = open('dataset/sc_liwc.dic', 'r')
    catagory = {}
    word_to_catagory = {}
    f.readline()
    while True:
        line = f.readline()
        if line[0] == '%':
            break
        str_num, str_cata = re.split('[ \t]', line)
        catagory[int(str_num)] = str_cata[:-2]
    while True:
        line = f.readline()
        if line == '':
            break
        tmp = re.split('[ \t]', line)
        word = tmp[0]
        if(word[-1] == '*'):
            word = word[:-1]
        word_to_catagory[word] = [string.atoi(str_num) for str_num in tmp[1:]]

    return catagory, word_to_catagory


if __name__ == '__main__':
    load_catagory()