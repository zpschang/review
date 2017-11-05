# coding: utf-8

# obtain original review data on douban

import urllib
import urllib2
import json
import time
from bs4 import BeautifulSoup
import os
import sys

cookie = open('cookie.txt').read()

str_to_point = {u'力荐': 5, u'推荐': 4, u'还行': 3, u'较差': 2, u'很差': 1}

tags = ['%E5%8F%AF%E6%92%AD%E6%94%BE',
        '%E7%BB%8F%E5%85%B8',
        '%E7%83%AD%E9%97%A8',
        '%E8%B1%86%E7%93%A3%E9%AB%98%E5%88%86',]
list_urls = ['https://movie.douban.com/j/search_subjects?type=movie&tag=%E5%8F%AF%E6%92%AD%E6%94%BE&sort=recommend&page_limit=20&page_start=',
                   'https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%BB%8F%E5%85%B8&sort=recommend&page_limit=20&page_start=',
                    ]

def get_movies():
    obj_movies = []
    for tag in tags:
        print 'tag:', tag
        for num in range(0, 1000, 20):
            print 'total movies:', len(obj_movies)
            try:
                list_url = 'https://movie.douban.com/j/search_subjects?type=movie&tag=' + tag + '&sort=recommend&page_limit=20&page_start='
                list_url += str(num)
                print 'url is:', list_url
                proxy = urllib2.ProxyHandler({'https': 'https://127.0.0.1:38251'})
                opener = urllib2.build_opener(proxy)
                urllib2.install_opener(opener)
                req = urllib2.Request(list_url)
                req.add_header('Cookie', cookie)
                resp = urllib2.urlopen(req).read()
                # print resp
                js = json.loads(resp)

                if len(js['subjects']) == 0:
                    break
                for json_obj in js['subjects']:
                    obj_movies.append(json_obj)
                time.sleep(0.1)

            except Exception:
                break


    file_movies = open('dataset/douban/movie_list.json', 'wb')
    file_movies.write(json.dumps(obj_movies))
    file_movies.close()

    return obj_movies

def get_comment():
    global program_id
    obj_movies = get_movies()
    movie_num = 0
    for obj in obj_movies:
        movie_num += 1
        if movie_num % 3 != program_id:
            continue
        movie_id = obj['id']
        movie_url = obj['url']
        movie_title = obj['title']
        filename = 'dataset/douban/comments/' + movie_id + '.json'
        obj_output = []
        start_index = 0
        try:
            file_comment = open(filename, 'rb')
            obj_output = json.load(file_comment)
            file_comment.close()
            start_index = len(obj_output)
            start_index = (start_index + 19) / 20 * 20 # continue retrieving comments
        except:
            pass
        for num in range(start_index, 2000, 20):
            try:
                comments_url = movie_url + 'comments?start=' + str(num) + '&limit=20&sort=new_score&status=P'
                print comments_url
                proxy = urllib2.ProxyHandler({'https': 'https://127.0.0.1:38251'})
                opener = urllib2.build_opener(proxy)
                urllib2.install_opener(opener)
                req = urllib2.Request(comments_url)
                req.add_header('Cookie', cookie)
                resp = urllib2.urlopen(req).read()

                soup = BeautifulSoup(resp, 'html5lib')
            except None:
                print 'num:', num
                print 'connection fail.'
                num -= 20
                time.sleep(15)
                break
            try:
                for comment in soup.find_all('div', class_='comment-item'):
                    point = -1
                    for span in comment.find_all('span'):
                        if span.get('title') in str_to_point:
                            point = str_to_point[span.get('title')]

                    comment_str = comment.find_all('p')[0].contents[0]
                    print point, comment_str.encode('utf-8'),
                    dict_output = {'comment_str': comment_str, 'point': point}
                    obj_output.append(dict_output)

            except IndexError:
                break
            print 'comment num:', num
            print 'movie: ', movie_title.encode('utf-8')
            print movie_num, '/', len(obj_movies)
            time.sleep(0.1)
        file_comment = open(filename, 'w')
        file_comment.write(json.dumps(obj_output))
        file_comment.close()

def get_info():
    obj_movies = get_movies()
    for obj in obj_movies:
        print 'retrieving info'
        movie_id = obj['id']
        movie_url = obj['url']
        title = obj['title']
        req = urllib2.Request(movie_url)
        req.add_header('Cookie', cookie)
        resp = urllib2.urlopen(req).read()
        soup = BeautifulSoup(resp, 'html5lib')
        intro = soup.find(id='link-report').contents[1].text
        genres = [genre.text for genre in soup.find_all('span', property='v:genre')]
        object = {'title': title, 'genre': genres, 'intro': intro}
        file_info = open('dataset/douban/movie_info/' + movie_id + '.json', 'w')
        json.dump(object, file_info)
        file_info.close()
        print 'title', title.encode('utf-8')
        print 'genre:',
        for genre in genres:
            print genre.encode('utf-8'),
        print '\n',
        print 'info:', intro.encode('utf-8')


if __name__ == '__main__':
    global program_id
    program_id = eval(sys.argv[1])
    print 'program_id:', program_id
    get_comment()