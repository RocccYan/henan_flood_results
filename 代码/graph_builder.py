# -*- coding: utf-8 -*-


import argparse
from utils import save_as_json, open_json, set_workspace, Logger


class GraphBuilder(object):
    """load raw data, and return a cleaned graph by edges."""

    def __init__(self, task_name):
        self.task_name = task_name
        self.logger = Logger(self.task_name).logger

    def build(self, path):
        self._edge_post(path)
        self._edge_attitude(path)
        self._edge_comment(path)
        self._edge_repost(path)
        self.build_graph(path)

        self.logger.info('raw edges loaded.')

    def _edge_post(self, path):
        """["45678876", Krer2L37]"""
        weibo_uid = open_json(f'{path}/weibo_uid.json')
        edge_post = list(map(lambda edge: [edge[1], edge[0]], weibo_uid))
        self.edge_post = self.edge_builder(edge_post)
        self.logger.info(f'built edge_post: {len(self.edge_post)} ')
        self.logger.info(f'built edge_post: {self.edge_post[0]} ')

    def _edge_repost(self, path):
        weibos_repost = open_json(f'{path}/weibo_reposts.json')
        edge_repost = list(map(lambda edge: [edge.get('weibo_id'), edge.get('repost_weibo_id')], weibos_repost))
        self.edge_repost = self.edge_builder(edge_repost)
        self.logger.info(f'built edge_repost: {len(self.edge_repost)} ')

    def _edge_attitude(self, path):
        weibos_attitude = open_json(f'{path}/weibo_attitudes.json')
        edge_attitude = list(map(lambda edge:[edge.get('attitude_uid'), edge.get('weibo_id')], weibos_attitude))
        self.edge_attitude = self.edge_builder(edge_attitude)
        self.logger.info(f'built edge_attitude: {len(self.edge_attitude)} ')

    def _edge_comment(self, path):
        weibos_comment = open_json(f'{path}/weibo_comments.json')
        edge_comment = list(map(lambda edge:[edge.get('comment_uid'), edge.get('weibo_id')], weibos_comment))
        self.edge_comment = self.edge_builder(edge_comment)
        self.logger.info(f'built edge_comment: {len(self.edge_comment)} ')

    def build_graph(self, path):
        graph = {
            'edge_post': self.edge_post,
            'edge_attitude': self.edge_attitude,
            'edge_comment': self.edge_comment,
            'edge_repost': self.edge_repost
        }
        save_as_json(graph, f'{path}/graph_{self.task_name}.json')

    @staticmethod
    def edge_builder(edges):
        drop_na = list(filter(lambda edge: edge[0] and edge[1], edges))
        drop_duplicate = set(map(lambda edge: edge[0] + '^^' + edge[1], drop_na))
        new_edges = list(map(lambda edge: [edge.split('^^')[0], edge.split('^^')[1]], drop_duplicate))
        return new_edges


if __name__ == '__main__':
    set_workspace()
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task_name', type=str, default='phase1')
    parser.add_argument('-p', '--path', type=str, default='./data/dataset_phase1')

    args = parser.parse_args()

    Builder = GraphBuilder(args.task_name)
    Builder.build(args.path)
