# -*- coding: utf-8 -*-

import time
from datetime import datetime
import os
import json
from utils import save_as_json, open_json, set_workspace, Logger


class DatasetSlicer(object):
    """
    slice some attribute we need.
    """

    def __init__(self, source=None):
        self.logger = Logger('DatasetSlicer').logger
        self.source = source
        self.dataset = open_json(self.source) if self.source else print('A Source is NEEDED.')

    def slice(self, attributes, target_path):
        if self.dataset:
            if isinstance(attributes, list):
                sliced_data = list(map(lambda item: tuple([item.get(attr, '') for attr in attributes]), self.dataset))
            else:
                sliced_data = list(map(lambda item: item.get(attributes), self.dataset))
            self.logger.info(f'Dataset {attributes} have been sliced.')
            save_as_json(sliced_data, target_path)
            self.logger.info(f'Dataset {attributes} saved at {target_path}.')

        else:
            print('Dataset NOT loaded.')
            self.logger.info('Dataset NOT loaded.')


if __name__ == '__main__':
    set_workspace()

    # user
    dataset = DatasetSlicer('./data/cleaned_data/user_infos.json')
    fields = ['uid', 'num_weibos', 'num_followings', 'num_followers', 'nickname',
              'gender', 'location', 'education', 'job', 'verification_reason', 'description',
              'tags']
    print(fields)
    dataset.slice(fields,
                       './data/results/user_attributes.json')

    # dataset_time = DatasetSlicer('./data/cleaned_data/weibos_split.json')
    # fields = ['weibo_id', 'num_repost', 'num_attitude', 'num_comment', 'is_origin', 'content', 'repost_weibo']
    # print(fields)
    # dataset_time.slice(fields,
    #                    './data/results/weibo_attributes.json')

    # -----------------------------------
    # weibo
    # dataset = open_json('./data/results/weibo_attributes.json')
    # data_demo = dataset[:100]
    # save_as_json(data_demo, './data/results/weibo_attributes_demo.json')

    print("Done.")
