# -*- coding: utf-8 -*-


import argparse
from datetime import datetime, timedelta
from utils import mkdir, open_json, save_as_json, Logger, set_workspace


class DatasetBuilder(object):
    """
    this aims to construct dataset by filters like time span.
    """

    def __init__(self, name):
        self.name = name
        self.target_path = f'./data/{name}'
        mkdir(self.target_path)
        self.logger = Logger('DatasetBuilder').logger

    def publish_time_filter(self, data, condition):
        try:
            begin = datetime.strptime(condition.get('value')[0], '%Y-%m-%d')
            end = datetime.strptime(condition.get('value')[1], '%Y-%m-%d') + timedelta(days=1)
            time = datetime.strptime(data.get('publish_time'), '%Y-%m-%d %H:%M')
            if begin <= time <= end:
                return True
            else:
                return False
        except Exception as e:
            self.logger.info(e, data)

    def building(self, condition, source=None):
        if not source:
            source = './data/cleaned_data'
        self.logger.info(f'Dataset rebuild from {source}')
        origin_weibo_data = open_json(f'{source}/weibos_split.json')

        if condition['type'] == 'publish_time':
            self.logger.info(f'Dataset filtering by {condition}...')
            weibo_data = list(filter(lambda item: self.publish_time_filter(item, condition), origin_weibo_data))
            save_as_json(weibo_data, f'{self.target_path}/weibos_split.json')
            weibo_uid = list(map(lambda x: (x.get('weibo_id'), x.get('uid')), weibo_data))
            save_as_json(list(weibo_uid), f'{self.target_path}/weibo_uid.json')
            weibo_id_base = set(map(lambda item: item.get('weibo_id'), weibo_data))
            save_as_json(list(weibo_id_base), f'{self.target_path}/weibo_id_base.json')
            self.logger.info(f'Dataset weibo_split rebuilt.')
            self.logger.info(f'Dataset weibo_split count: {len(weibo_id_base)}.')

            origin_attitude_data = open_json(f'{source}/weibo_attitudes.json')
            attitude_data = list(filter(lambda item: item.get('weibo_id') in weibo_id_base, origin_attitude_data))
            save_as_json(attitude_data, f'{self.target_path}/weibo_attitudes.json')
            self.logger.info(f'Dataset weibo_attitudes rebuilt.')
            self.logger.info(f'Dataset weibo_attitudes count: {len(attitude_data)}.')

            origin_repost_data = open_json(f'{source}/weibo_reposts.json')
            repost_data = list(filter(lambda item: item.get('weibo_id') in weibo_id_base and
                                                   item.get('repost_weibo_id') in weibo_id_base, origin_repost_data))
            save_as_json(repost_data, f'{self.target_path}/weibo_reposts.json')
            self.logger.info(f'Dataset weibo_reposts rebuilt.')
            self.logger.info(f'Dataset weibo_reposts count: {len(repost_data)}.')

            origin_comment_data = open_json(f'{source}/weibo_comments.json')
            comment_data = list(filter(lambda item: item.get('weibo_id') in weibo_id_base, origin_comment_data))
            save_as_json(comment_data, f'{self.target_path}/weibo_comments.json')
            self.logger.info(f'Dataset weibo_comments rebuilt.')
            self.logger.info(f'Dataset weibo_comments count: {len(comment_data)}.')

            self.logger.info(f'Dataset {self.name} rebuilt successfully!')


if __name__ == '__main__':
    set_workspace()
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task_name', type=str, default='dataset_phase1')
    parser.add_argument('-e', '--end', type=str, default='2021-07-20')
    args = parser.parse_args()

    data0901 = DatasetBuilder(args.task_name)
    timespan = {'type': 'publish_time', 'value': ('2021-07-10', args.end)}
    print(timespan)
    data0901.building(timespan, )
