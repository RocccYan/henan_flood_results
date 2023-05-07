# -*- coding: utf-8  -*-


import json
import scrapy
import re
import time
import requests
import random
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from lxml import html
from crawler_wb.items import CrawlerWbItem, CrawlerWbCommentItem, CrawlerWbAttitudeItem, CrawlerWbRepostItem, \
    CrawlerWbUserItem
from crawler_wb.configs.configs import configs_zju as configs
from crawler_wb.utils import weiboID_encoder


os.environ['NO_PROXY'] = 'weibo.cn'


class WeiboSpider(scrapy.Spider):
    name = 'weibo'
    allowed_domains = ['weibo.cn']
    # cookies_raw = configs['cookies']
    # cookies = {cookie.split('=')[0]: cookie.split('=')[-1] for cookie in cookies_raw.split('; ')}
    start_urls = configs['start_urls']
    search = configs.get("search", '')
    weiboID_crawled = defaultdict(lambda: defaultdict(set))
    weibo_crawled_path = configs.get("weibo_crawled_path", './weibo_crawled.txt')
    user_agent = configs['user-agent']
    uid_to_crawl_path = configs.get("uid_to_crawl_path", './uid_to_crawl.txt')
    decoder = weiboID_encoder()

    def __init__(self, uid=None, org=None):
        super(WeiboSpider).__init__()
        self.uid_crawled_path = configs['uid_crawled_path']
        self.uid_all_path = configs['uid_all_path']
        self._keyword_list = configs.get('keyword_list', [])

        self.cookies_raw = configs['cookies_own']
        # self.proxy = configs['proxy'][group]
        # self.logger.info(f'seed_uid:{self.seed_uid_list}')

        self.begin_date = configs.get('begin_date', '0')
        self.weibo_id_crawled_path = configs.get('weibo_id_crawled_path')
        self.weibo_id_crawled = set()
        # self.sortByhot = True

    def get_uid_to_crawl(self):
        with open(self.uid_to_crawl_path, 'r', encoding='utf-8') as fh:
            uid_to_crawl = set(fh.readlines())
        try:
            with open(self.uid_crawled_path, 'r', encoding='utf-8') as fh:
                uid_crawled = set(fh.readlines())
            uid_to_crawl_now = uid_to_crawl - uid_crawled
        except Exception as e:
            self.logger.exception(e)
            uid_to_crawl_now = uid_to_crawl
        with open(self.weibo_id_crawled_path, 'a+', encoding='utf-8') as fh:
            self.weibo_id_crawled = set(fh.readlines())
        return uid_to_crawl_now

    def get_random_cookies(self):
        cookies = {cookie.split('=')[0]: cookie.split('=')[-1] for cookie in
                   random.choice(self.cookies_raw).split('; ')}
        self.logger.info(f'Requests with cookies: {cookies.get("SSOLoginState")}')
        return cookies

    def start_requests(self):
        self.seed_uid_list = self.get_uid_to_crawl()
        for uid_url in self.seed_uid_list:
            if uid_url and self.assignment(uid_url):
                self.logger.info(f'Homepage:{uid_url}')
                yield scrapy.FormRequest(url=uid_url, cookies=self.get_random_cookies(), callback=self.parse,
                                         # meta={"proxy": "http://{}".format(proxy)}
                                         )
            else:
                self.logger.info(f"start_requests {uid_url} run failed.")

    def get_homepage(self, seed_uid, is_org):
        self.logger.info(f'{sys._getframe().f_code.co_name}:{seed_uid}')
        if is_org:
            url = f"{self.start_urls['org_home_page']}{seed_uid}"
        else:
            url = f"{self.start_urls['personal_home_page']}{seed_uid}"
        return url

    def assignment(self, url):
        if configs.get('need_assignment'):
            try:
                go = int(url.strip()[-1]) % 2
                if go:
                    return False
                else:
                    return True
            except:
                return False
        else:
            return True

    def parse(self, response):

        # 1. get response
        self.logger.info(f"Parsing Begins......")
        uid, nickname = self._get_uid_personal_homepage(response)
        self.logger.info(f"Get uid,nicknames:{uid, nickname}")
        user_info_surl = response.xpath("//div[@class='u']//a[contains(@href,'info')]/@href").get()
        if user_info_surl:
            user_info_url = self.start_urls['root'] + user_info_surl if not 'http' in user_info_surl else user_info_surl
            nums_info = response.xpath("//div[@class='tip2']//text()").getall()
            for num in nums_info:
                if '微博' in num:
                    num_weibos = num
                elif '关注' in num:
                    num_followings = num
                elif '粉丝' in num:
                    num_followers = num

            yield scrapy.Request(url=user_info_url, callback=self.parse_user_info, cookies=self.get_random_cookies(),
                                 meta={'uid': uid, 'num_weibos': num_weibos, 'num_followings': num_followings,
                                       'num_followers': num_followers})
        # yield filters
        url_search = self._get_url_search(response, uid)
        for url in url_search:
            yield scrapy.Request(url=url, callback=self.parse_weibo_searched, cookies=self.get_random_cookies(),
                                 meta={"uid": uid, "nickname": nickname, "search_url": url})

    def parse_weibo_searched(self, response):
        uid = response.meta.get("uid", '')
        nickname = response.meta.get("nickname", '')
        search_url = response.meta.get("search_url")
        weibo_items = response.xpath("//div[@class='c'][contains(@id,'M_')]")

        for weibo in weibo_items:

            weibo_id_info = weibo.xpath("./@id").get()
            weibo_id = weibo_id_info.split('_')[1]

            is_new_weibo = self.is_new_weibo(weibo_id)
            if is_new_weibo:
                self.logger.info(f'Counter: new weibo of {uid} recorded.')
                yield from self.parse_single_weibo(weibo, uid, nickname, weibo_id)

        # Next Page
        short_url_next_page = response.xpath("//a[text()='下页']/@href").get()
        if short_url_next_page:
            url_next_page = response.urljoin(short_url_next_page)
            print(f'---------------Weibo Next page\n{url_next_page}\n----------------------')
            self.logger.info(f'---------------Weibo Next page\n{url_next_page}\n----------------------')
            yield scrapy.Request(url=url_next_page, callback=self.parse_weibo_searched,
                                 cookies=self.get_random_cookies(),
                                 meta={"uid": uid, "nickname": nickname, "search_url": search_url}
                                 )
        else:
            self.logger.info(f'All searched weibo by {search_url} has been crawled: {uid}')
            # self.write_uid_crawled(uid, search_url)
            # self.write_weibo_id_crawled(uid)
            try:
                del self.weiboID_crawled[uid][search_url]
            except:
                pass

    def write_uid_crawled(self, uid, search_url=None):
        with open(self.uid_crawled_path, 'a+', encoding='utf-8') as fh:
            fh.write(f'{self.start_urls.get("personal_home_page")}{uid}' + '\n')
            fh.write(f'{self.start_urls.get("org_home_page")}{uid}' + '\n')
            fh.write(search_url + '\n')

    def is_new_weibo_by_search(self, uid, weibo_id, search_url):
        self.logger.info('In is_new_weibo')
        is_new_weibo = True
        weibo_dict = self.weiboID_crawled.get(uid, '')
        if weibo_dict:
            all_weibo = set()
            for values in weibo_dict.values():
                all_weibo.update(values)
            if weibo_id in all_weibo:
                is_new_weibo = False
        else:
            self.weiboID_crawled[uid][search_url].add(weibo_id)
        return is_new_weibo

    def is_new_weibo(self, weibo_id):
        self.logger.info('In is_new_weibo')
        is_new_weibo = True
        if weibo_id in self.weibo_id_crawled:
            return False
        else:
            self.weibo_id_crawled.update(weibo_id)
            with open(self.weibo_id_crawled_path, 'a+', encoding='utf-8') as fh:
                fh.write(weibo_id + '\n')
        return is_new_weibo

    def _get_url_search(self, response, uid):
        url_search = []
        search_starttime = self.search.get("starttime", "")
        search_endtime = self.search.get("endtime", "")
        url_root = f'{self.start_urls.get("root")}/{uid}'
        self.logger.info(f'Search URL root: {url_root}')
        for keyword in self.search.get("keywords", []):
            _url_search = f"{url_root}/profile?keyword={keyword}&hasori=0&haspic=0&starttime={search_starttime}&endtime={search_endtime}&advancedfilter=1&page=1"
            url_search.append(_url_search)
            self.logger.info(f'Search URL url_search: {_url_search}')
        # os._exit(-1)
        return url_search

    def parse_comment(self, response):

        weibo_id = response.meta.get("weibo_id", "")
        self.logger.info('Parse comment Begins.....')
        # 1. get response
        node_comments = response.xpath("//div[contains(@id,'C_')]")
        for cmt in node_comments:
            self.logger.info(f'Counter: comment recorded.')
            item = CrawlerWbCommentItem()
            href = response.xpath("//div[@class='n']//a[contains(text(),'刷新')]/@href").get()
            item['weibo_id'] = weibo_id
            item['comment_id'] = cmt.xpath("./@id").get()
            href_uid = cmt.xpath("./a[contains(@href,'fuid')]/@href").get()
            item['comment_uid'] = self._get_comment_uid(href_uid)
            item['comment_content'] = cmt.xpath("./span[@class='ctt']/text()").get()
            # print(cmt.xpath(".//a[contains(text(),'赞')]/text()"))
            item['num_comment_attitude'] = cmt.xpath(".//a[contains(text(),'赞')]/text()").get()
            item['comment_time'] = self._get_publish_time(cmt)
            self.logger.info(f"{sys._getframe().f_code.co_name}:{item}")
            yield item

        # Next Page
        # part_url = response.xpath("//a[text()='下页']/@href").get()
        # if part_url:
        #     next_page = response.urljoin(part_url)
        #     # construct request
        #     # proxy = get_proxy().get("proxy")
        #     yield scrapy.Request(url=next_page, callback=self.parse_comment, cookies=self.get_random_cookies(),
        #                          meta={"weibo_id": weibo_id}
        #                          )
        #     print(f'-----------COMMENT, Next Page{next_page}------------\n')

    def need_repair(self, url_roll):
        need = False
        uid = url_roll.split('/')[-1]
        if not uid in self.weiboID_crawled.keys():
            with open(self.uid_crawled_path, 'r', encoding='utf-8') as fh:
                uid_crawled = set(fh.readlines())
            if not url_roll in uid_crawled:
                need = True
        return need

    def get_pagelist(self, weibo_id, type='', num=0):
        pagelist = []
        num = int(num) if len(num) > 0 else 0
        if type != 'reposts':
            num = min(num, 50)
        if num:
            pagelist = [f"{self.start_urls[type]}{weibo_id}?&page={n}" for n in range(1, int(num/10)+1+1)]
        return pagelist

    def parse_single_weibo(self, weibo, uid, nickname, weibo_id):

        item = CrawlerWbItem()
        item['weibo_id'] = weibo_id
        item['time_crawl'] = time.strftime("%Y%m%d%H%M%S", time.localtime())
        item['uid'] = uid
        item['user_name'] = nickname
        item['tag'] = weibo.xpath(".//a[contains(text(),'#')]/text()").getall()

        # item['content'] = weibo.xpath(".//span[@class='ctt']/text()").get()
        item['content'] = self.get_weibo_content(weibo_id)
        time_created_source = weibo.xpath(".//span[@class='ct']/text()").get().split(u"\xa0")
        item['source'] = time_created_source[1] if len(time_created_source) > 1 else 'No source'

        num_repost = self._get_nums_interaction(
            weibo.xpath(".//a[contains(@href, 'weibo.cn/repost')]/text()").get())
        num_comment = self._get_nums_interaction(
            weibo.xpath(".//div[last()]/a[contains(@href, 'weibo.cn/comment')]/text()").get())
        num_attitude = self._get_nums_interaction(
            weibo.xpath(".//a[contains(@href, 'weibo.cn/attitude')]/text()").get())

        item['num_repost'] = num_repost
        item['num_comment'] = num_comment
        item['num_attitude'] = num_attitude

        # yield request on attitudes, reposts, comments
        url_comment_list = self.get_pagelist(weibo_id, type='comments', num=num_comment)
        url_attitude_list = self.get_pagelist(weibo_id, type='attitudes', num=num_attitude)
        url_repost_list = self.get_pagelist(weibo_id, type='reposts', num=num_repost)

        self.logger.info(f"num_repost in parse_single_weibo:{num_repost}")
        self.logger.info(f"url_repost_list in parse_single_weibo:{url_repost_list}")

        for url_comment in url_comment_list:
            yield scrapy.Request(url=url_comment, callback=self.parse_comment, cookies=self.get_random_cookies(),
                                 meta={"weibo_id": weibo_id}
                                 )
            print(f'-----------COMMENT, Next Page{url_comment}------------\n')
        for url_attitude in url_attitude_list:
            yield scrapy.Request(url=url_attitude, callback=self.parse_attitude, cookies=self.get_random_cookies(),
                                 meta={"weibo_id": weibo_id}
                                 )
            print(f'-----------Attitude, Next Page:{url_attitude}------------\n')
        for url_repost in url_repost_list:
            yield scrapy.Request(url=url_repost, callback=self.parse_repost, cookies=self.get_random_cookies(),
                                 meta={"weibo_id": weibo_id}
                                 )
            print(f'-----------repost, Next Page:{url_repost}------------\n')

        # crawl weibos of the user who post the origin weibo
        _repost = weibo.xpath(".//div/span[@class='cmt']")
        item['is_origin'] = False if len(_repost) >= 3 else True
        if not item['is_origin']:
            url_origin_weibo_user = _repost.xpath(".//a/@href").get()
            url_roll = self._get_url_origin_user(url_origin_weibo_user)
            self.logger.info(f'Parse single weibo: {url_roll}')
            if url_roll:
                new_url = self._add_uid_to_crawl(url_roll)
                if new_url and self.assignment(url_roll):
                    yield scrapy.Request(url=url_roll, callback=self.parse, cookies=self.get_random_cookies(),
                                         # meta={"proxy": "http://{}".format(proxy)}
                                         )
                else:
                    need_repair = self.need_repair(url_roll)
                    if need_repair and self.assignment(url_roll):
                        yield scrapy.Request(url=url_roll, callback=self.parse, cookies=self.get_random_cookies(),
                                             # meta={"proxy": "http://{}".format(proxy)}
                                             )
        item['publish_tool'] = self._get_publish_tool(weibo)
        item['publish_place'] = self._get_publish_place(weibo)
        item['publish_time'] = self._get_publish_time(weibo)
        if item['publish_time'] < self.begin_date:
            self.logger.info(f'Crawling stopped at {item["publish_time"]} of {item["uid"]}.')
            return None
        # item['time_created'] = timeCreated_source[0]
        self.logger.info(f'weibo to be yielded: {item}')
        yield item

    def get_weibo_content(self, weibo_id):
        url_weibo_content = f"{self.start_urls.get('comments')}{weibo_id}"
        self.logger.info(f' request with: {url_weibo_content}')
        cookies = random.choice(self.cookies_raw)
        user_agent = self.user_agent
        headers = {"Cookie": cookies,
                   "User-Agent": user_agent}
        resp = requests.get(url=url_weibo_content,
                            headers=headers
                            )
        resp.encoding = resp.apparent_encoding
        try:
            tree = html.fromstring(resp.content)
            weibo_content = tree.xpath("string(//div[@id='M_'])")
        except:
            weibo_content = f'Content of {weibo_id} lost'
        return weibo_content

    def _get_url_origin_user(self, url_origin_weibo_user):
        url_roll = ""
        if url_origin_weibo_user:
            if not "http" in url_origin_weibo_user:
                url_roll = self.start_urls.get("root") + url_origin_weibo_user
        self.logger.info(f'{sys._getframe().f_code.co_name}:{url_roll}')
        return url_roll

    # def _is_skip(self, weibo):
    #     """filter, whether skip this weibo"""
    #     is_skip = True
    #     content = weibo.xpath(".//span[@class='ctt']/text()").get()
    #     tags = weibo.xpath(".//a[contains(text(),'#')]/text()").getall()
    #     for keyword in self._keyword_list:
    #         if isinstance(keyword, tuple):
    #             is_skip = not min([word in content for word in keyword])
    #             return is_skip
    #         elif keyword in content:
    #             is_skip = False
    #             return is_skip
    #
    #     for keyword in self._keyword_list:
    #         if isinstance(tags, list):
    #             tags = "".join(tags)
    #         if isinstance(keyword, tuple):
    #             is_skip = not min([word in tags for word in keyword])
    #             return is_skip
    #         elif keyword in tags:
    #             is_skip = False
    #             return is_skip
    #     return is_skip

    def parse_attitude(self, response):

        self.logger.info('Parse attitudes Begins.....')
        # 1. get response
        # node_attitudes = response.xpath("//div[@class='c']/a[contains(@href,'/u/')]/..")
        node_attitudes = response.xpath("//div[@class='c' and a[contains(@href,'/u/')] and span[@class='ct']]")
        # attitude_uids = response.xpath("//div[@class='c']/a[contains(@href,'/u/')]//a[contains(@href,'/u/')]/@href")
        weibo_id = response.meta.get('weibo_id', '')
        self.logger.info(f'weibo_id in parse attitude.:{weibo_id}')
        for attitude in node_attitudes:
            self.logger.info(f'Counter: attitude recorded.')
            item = CrawlerWbAttitudeItem()
            item['weibo_id'] = weibo_id
            print('ATTITUDE_UID--------------------------')
            print(attitude.xpath("//a[contains(@href,'/u/')]/@href").get())
            item['attitude_uid'] = self._get_uid_weibo(attitude.xpath("//a[contains(@href,'/u/')]/@href").get())
            item['attitude_time'] = self._get_publish_time(attitude)
            self.logger.info(f"{sys._getframe().f_code.co_name}:{item}")
            yield item

        # Next Page
        # part_url = response.xpath("//a[text()='下页']/@href").get()
        # if part_url:
        #     next_page = response.urljoin(part_url)
        #     # construct request
        #     # proxy = get_proxy().get("proxy")
        #     yield scrapy.Request(url=next_page, callback=self.parse_attitude, cookies=self.get_random_cookies(),
        #                          meta={"weibo_id": weibo_id}
        #                          )
        #     print(f'-----------Attitude, Next Page:{next_page}------------\n')

    def parse_repost(self, response):

        self.logger.info(f'Parse reposts Begins.....:{response}')
        # 1. get response
        # node_reposts = response.xpath("//div[@class='c']//span[@class='cc']/a/@href")
        node_reposts = response.xpath("//div[@class='c']//span[@class='cc']/..")
        repost_weibo_ids = response.xpath("//div[@class='c']//span[@class='cc']/a/@href")
        weibo_id = response.meta.get('weibo_id', '')
        for repost, repost_weibo_id in zip(node_reposts, repost_weibo_ids):
            self.logger.info(f'Counter: repost recorded.')
            item = CrawlerWbRepostItem()
            item['weibo_id'] = weibo_id
            item['repost_weibo_id'] = self._get_weiboID_repost(repost_weibo_id.get())
            # item['repost_weibo_id'] = self._get_weiboID_repost(repost.xpath("//span[@class='cc']/a/@href").get())
            item['repost_time'] = self._get_publish_time(repost)
            self.logger.info(f'repost to be yielded: {item}')
            yield item

        repost_uid = response.xpath("//div[@class='c']//span[@class='cc']/../a[contains(@href,'/u/')]/@href")
        for uid in repost_uid:
            url_roll = self._get_url_origin_user(uid.get())
            self.logger.info(f'Parse repost:{url_roll}')
            if url_roll:
                new_url = self._add_uid_to_crawl(url_roll)
                if new_url and self.assignment(url_roll):
                    yield scrapy.Request(url=url_roll, callback=self.parse, cookies=self.get_random_cookies(),
                                         # meta={"proxy": "http://{}".format(proxy)}
                                         )
                else:
                    need_repair = self.need_repair(url_roll)
                    if need_repair and self.assignment(url_roll):
                        yield scrapy.Request(url=url_roll, callback=self.parse, cookies=self.get_random_cookies(),
                                             # meta={"proxy": "http://{}".format(proxy)}
                                             )
        # Next Page
        # part_url = response.xpath("//a[text()='下页']/@href").get()
        # if part_url:
        #     next_page = response.urljoin(part_url)
        #     # construct request
        #     # proxy = get_proxy().get("proxy")
        #     yield scrapy.Request(url=next_page, callback=self.parse_repost, cookies=self.get_random_cookies(),
        #                          meta={"weibo_id": weibo_id}
        #                          )
        #     print(f'-----------repost, Next Page:{next_page}------------\n')

    def parse_repost_nospread(self, response):

        self.logger.info(f'Parse reposts Begins.....:{response}')
        # 1. get response
        node_reposts = response.xpath("//div[@class='c']//span[@class='cc']/a/@href")
        # self.logger.info(f"{sys._getframe().f_code.co_name}:node_reposts:{node_reposts}")
        weibo_id = response.meta.get('weibo_id', '')
        for repost in node_reposts:
            self.logger.info(f'Counter: repost recorded.')
            item = CrawlerWbRepostItem()
            item['weibo_id'] = weibo_id
            item['repost_weibo_id'] = self._get_weiboID_repost(repost.get())
            self.logger.info(f'repost to be yielded: {item}')
            yield item

        # Next Page
        part_url = response.xpath("//a[text()='下页']/@href").get()
        if part_url:
            next_page = response.urljoin(part_url)
            # construct request
            # proxy = get_proxy().get("proxy")
            yield scrapy.Request(url=next_page, callback=self.parse_repost, cookies=self.get_random_cookies(),
                                 meta={"weibo_id": weibo_id}
                                 )
            print(f'-----------repost, Next Page:{next_page}------------\n')

    def _get_publish_time(self, info):
        """获取微博发布时间"""
        try:
            str_time = info.xpath(".//span[@class='ct']/text()").get()
            # str_time = self._handle_garbled(str_time[0])
            self.logger.info(f'{sys._getframe().f_code.co_name},str_time:{str_time}')
            publish_time = str_time.split(u'来自')[0]
            if u'刚刚' in publish_time:
                publish_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            elif u'分钟' in publish_time:
                minute = publish_time[:publish_time.find(u'分钟')]
                minute = timedelta(minutes=int(minute))
                publish_time = (datetime.now() -
                                minute).strftime('%Y-%m-%d %H:%M')
            elif u'今天' in publish_time:
                today = datetime.now().strftime('%Y-%m-%d')
                time = re.search('[\d]{1,2}:[\d]{1,2}', publish_time)[0]
                publish_time = today + ' ' + time
                if len(publish_time) > 16:
                    publish_time = publish_time[:16]
            elif u'月' in publish_time:
                year = datetime.now().strftime('%Y')
                # month = publish_time[0:2]
                month = re.search('([\d]{1,2})月', publish_time)[1]
                day = re.search('([\d]{1,2})日', publish_time)[1]
                time = re.search('[\d]{1,2}:[\d]{1,2}', publish_time)[0]
                publish_time = year + '-' + month + '-' + day + ' ' + time
            else:
                publish_time = publish_time[:16]
            self.logger.info(f"{sys._getframe().f_code.co_name}:{publish_time}")
            return publish_time
        except Exception as e:
            self.logger.exception(e)

    def write_weibo_id_crawled(self, uid):
        with open(self.weibo_crawled_path, 'a+', encoding='utf-8') as fh:
            fh.write(f'{uid}: {self.weiboID_crawled.get(uid, "")}\n')

    def _add_uid_to_crawl(self, uid):
        new_uid = False
        try:
            with open(self.uid_to_crawl_path, 'a+', encoding='utf-8') as fh:
                uids = set(fh.readlines())
                if uid not in uids:
                    fh.write(str(uid) + '\n')
                    new_uid = True
        except Exception as e:
            self.logger.exception(f"Record uid crawled failed:{e}")
        return new_uid

    def _get_weibo_content(self, response):
        content = ''
        try:
            content = response.xpath("string(//div[@id='M_'])")
        except Exception as e:
            self.logger.info(f'Weibo Content Not Found: {e}')
        return content

    def _get_uid_personal_homepage(self, response):
        uid, nickname = "", ""
        try:
            uid_info = response.xpath("//div[@class='u']//a[contains(@href,'follow')]/@href").get()
            self.logger.info(f'uid_info:{uid_info}')
            uid = self._get_uid_follow(uid_info)
            self.logger.info(f'uid:{uid}')
            # os._exit(-1)
            nickname_info = response.xpath("//div[@class='u']//div[@class='ut']//span[@class='ctt']/text()")
            if nickname_info:
                nickname = nickname_info.get()
            else:
                nickname_info = response.xpath("//div[@class='u']//div[@class='ut']/text()")
                nickname = nickname_info.get()
            return uid, nickname
        except Exception as e:
            self.logger.exception(f'Get uid failed:{e}')
            return uid, nickname

    def parse_user_info(self, response):

        item = CrawlerWbUserItem()
        item['uid'] = response.meta.get('uid', '')
        item['num_weibos'] = response.meta.get('num_weibos', '')
        item['num_followings'] = response.meta.get('num_followings', '')
        item['num_followers'] = response.meta.get('num_followers', '')
        self.logger.info(f"User Info Uid: {item['uid']}")

        basic_info = response.xpath("//div[@class='c'][contains(text(),'昵称')]/text()").getall()
        for info in basic_info:
            if '昵称' == info[:2]:
                item['nickname'] = info[3:]
            elif '认证信息' == info[:4]:
                item['verification_reason'] = info[5:]
            elif '性别' == info[:2]:
                item['gender'] = info[3:]
            elif '地区' == info[:2]:
                item['location'] = info[3:]
            elif '生日' == info[:2]:
                item['birthday'] = info[3:]
            elif '简介' == info[:2]:
                item['description'] = info[3:]
        info_blocks = response.xpath("//div[@class='tip']/text()").getall()
        if "学习经历" == info_blocks[1]:
            item['education'] = response.xpath("html/body/div[8]").get()
            if "工作经历" == info_blocks[2]:
                item['job'] = response.xpath("html/body/div[10]").get()
        elif "工作经历" == info_blocks[1]:
            item['job'] = response.xpath("html/body/div[8]").get()

        tag_info = response.xpath("//div[@class='c'][contains(text(),'昵称')]/a/text()").getall()
        item['tags'] = tag_info
        yield item

    def _get_nickname(self, href):
        """get weiboID from follow href."""
        try:
            return re.match('/([0-9a-zA-Z]+)?.*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _get_uid_follow(self, href):
        """get weiboID from follow href."""
        try:
            return re.match('/(\d+)?.*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _get_weiboID_repost(self, href):
        """get weiboID which repost the weibo in this page."""
        # //div[@class='c']//span[@class='cc']/a/@href
        try:
            return re.match('/attitude/([0-9a-zA-Z]+)?.*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _get_publish_tool(self, response):
        tool = 'None'
        try:
            text = response.xpath("//span[@class='ct']/text()").get()
            tool = text.split(u'来自')[1]
            return tool
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return tool

    def _get_publish_place(self, response):
        try:
            place = response.xpath(".//a[contains(@href,'sinaurl')]/text()").get()
            return place
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _get_comment_uid(self, url):
        if not url:
            return None
        pattern = ".*fuid=(\d+)?&.*"
        return re.match(pattern, url)[1]

    def _get_uid_weibo(self, href):
        uid = None
        try:
            uid = re.match('.*?/u/(\d+).*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
        return uid

    def _get_raw_html(self, response, page_name):
        with open(f'{page_name}.html', 'wb', encoding='utf-8') as f:
            f.write(response.body)

    def _get_weiboID_comment(self, href):
        try:
            return re.match('/comment/([0-9a-zA-Z]+)?.*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _get_weiboID_attitude(self, href):
        weiboID = None

        try:
            weiboID = re.match('/attitude/([0-9a-zA-Z]+)?.*', href)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
        return weiboID

    def _get_nums_interaction(self, text):
        try:
            return re.match('.*?(\d+).*', text)[1]
        except Exception as e:
            self.logger.info(f"{sys._getframe().f_code.co_name}:{e}")
            return ""

    def _handle_garbled(self, info):
        """处理乱码"""
        try:
            self.logger.info(f'{sys._getframe().f_code.co_name}:{info}')
            # info = (info.xpath('string(.)').replace(u'\u200b', '').encode(
            info = (info.replace(u'\u200b', '').encode(
                sys.stdout.encoding, 'ignore').decode(sys.stdout.encoding))
            self.logger.info(f'{sys._getframe().f_code.co_name}_after:{info}')
            return info
        except Exception as e:
            self.logger.info(e)
            return u'无'


def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))


if __name__ == '__main__':
    spider = WeiboSpider()
