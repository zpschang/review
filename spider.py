# coding: utf-8

import urllib
import urllib2
import json
import time
from bs4 import BeautifulSoup
import os

obj_movies = []
str_to_point = {u'力荐': 5, u'推荐': 4, u'还行': 3, u'较差': 2, u'很差': 1}

for num in range(0, 1000, 20):
    print num
    try:
        list_url = 'https://movie.douban.com/j/search_subjects?type=movie&tag=%E5%8F%AF%E6%92%AD%E6%94%BE&sort=recommend&page_limit=20&page_start='
        list_url += str(num)
        req = urllib2.Request(list_url)
        resp = urllib2.urlopen(req).read()
        js = json.loads(resp)

        if len(js['subjects']) == 0:
            break
        for json_obj in js['subjects']:
            obj_movies.append(json_obj)
        time.sleep(1)

    except Exception:
        break


file_movies = open('dataset/douban/movie_list.json', 'w')
file_movies.write(json.dumps(obj_movies))
file_movies.close()

for obj in obj_movies:
    movie_id = obj['id']
    movie_url = obj['url']
    obj_output = []
    for num in range(0, 1000, 20):
        try:
            comments_url = movie_url + 'comments?start=' + str(num) + '&limit=20&sort=new_score&status=P'
            req = urllib2.Request(comments_url)
            req.add_header('Cookie', 'bid=L9HzO_vISEw; gr_user_id=df97a5fa-2d83-46dc-8fa2-349e8e8242cf; __yadk_uid=t7kiVznpnGbqfYRIGs2knRtOJAHymky6; viewed="1958285_1422882_26253943_25742296"; ll="108090"; _ga=GA1.2.2107204329.1499661793; _gid=GA1.2.1326077005.1505626177; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1505637941%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D; _vwo_uuid_v2=59495F45CBFC26B16423CB2090DB073B|acdaf5ce5a21f1a4d8448c6a915622d9; __utmt=1; ps=y; ap=1; dbcl2="151744674:pOGRdk5xsKI"; ck=zP4M; _pk_id.100001.4cf6=7711defdeb261e8b.1503127256.6.1505638772.1505634859.; _pk_ses.100001.4cf6=*; __utma=30149280.2107204329.1499661793.1505632230.1505637941.21; __utmb=30149280.6.10.1505637941; __utmc=30149280; __utmz=30149280.1505625661.19.17.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmv=30149280.15174; __utma=223695111.1558003340.1503127256.1505632230.1505637941.7; __utmb=223695111.0.10.1505637941; __utmc=223695111; __utmz=223695111.1505626183.5.5.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; push_noty_num=0; push_doumail_num=0')
            resp = urllib2.urlopen(req).read()
            soup = BeautifulSoup(resp, 'html5lib')
        except:
            print 'num:', num
            print 'connection fail.'
            num -= 20
            time.sleep(15)
            continue
        for comment in soup.find_all('div', class_='comment-item'):
            point = -1
            for span in comment.find_all('span'):
                if span.get('title') in str_to_point:
                    point = str_to_point[span.get('title')]

            comment_str = comment.find_all('p')[0].contents[0]
            print point, comment_str
            dict_output = {'comment_str': comment_str, 'point': point}
            obj_output.append(dict_output)

        print 'comment num:', num
        time.sleep(0.1)
    file_comment = open('dataset/douban/comments/' + movie_id + '.json', 'w')
    file_comment.write(json.dumps(obj_output))
    file_comment.close()