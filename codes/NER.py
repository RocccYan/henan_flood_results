#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import math
import time
from multiprocessing import Pool

from transformers import AutoModelForTokenClassification,AutoTokenizer,pipeline
from utils import save_as_json, open_json, Logger, set_workspace, mkdir


class NER(object):

    def __init__(self, model_name='multilingual-ner'):
        """
        loading model and tokenizer.
        :param model: option model includes: "clue-ner", "multilingual-ner", "ckip-ner"
        """
        self.model_name = model_name
        if model_name == 'clue-ner':
            model = AutoModelForTokenClassification.from_pretrained('./models/roberta-base-finetuned-cluener2020-chinese')
            tokenizer = AutoTokenizer.from_pretrained('./models/roberta-base-finetuned-cluener2020-chinese')
        elif model_name == 'multilingual-ner':
            tokenizer = AutoTokenizer.from_pretrained("./models/bert-base-multilingual-cased-ner-hrl")
            model = AutoModelForTokenClassification.from_pretrained("./models/bert-base-multilingual-cased-ner-hrl")
        elif model_name == 'ckip-ner':
            tokenizer = AutoTokenizer.from_pretrained(".models/bert-base-chinese-ner")
            model = AutoModelForTokenClassification.from_pretrained(".models/bert-base-chinese-ner")

        self.ner = pipeline("ner", model=model, tokenizer=tokenizer)

    def get_ner_results(self, text, formatted=True):
        """
        :param text:
        :return: ner result in (word, tag, score)
        """
        if isinstance(text, str):
            if self.model_name == 'clue-ner':
                length_text = len(text)
                inner_batch = length_text // 500
                if inner_batch:
                    results = []
                    i = 0
                    while i <= inner_batch:
                        results.extend(self.ner(text[i * 500:(i + 1 * 500)], aggregation_strategy='simple'))
                        i += 1
                else:
                    results = self.ner(text, aggregation_strategy='simple')
            else:
                results = self.ner(text, aggregation_strategy='simple')

            if formatted:
                new_results = list(map(self.get_simple_result, results))
                return new_results
            else:
                return results
        else:
            return []

    @staticmethod
    def get_simple_result(item):
        simple_item = tuple([
                        item.get('entity_group'),
                        item.get('word').replace(' ', ''),
                        str(round(item.get('score'), 4))])
        return simple_item


def get_content_text(datatype):
    """
    context will be saved with weibo_id in a tuple, i.e. {weibo_id, content}
    :param datatype:
    :return:
    """

    if datatype == 'weibo':
        data = open_json('./data/cleaned_data/weibos_split.json')
        data_contents = [(item.get('weibo_id'), item.get('content', '')) for item in data]
    elif datatype == 'comment':
        data = open_json('./data/cleaned_data/weibo_comments.json')
        data_contents = [(item.get('comment_id'), item.get('comment_content', '')) for item in data]

    save_as_json(data_contents, f'./data/cleaned_data/{datatype}_contents.json')


def get_subset(filepath, num_subset):
    data = open_json(filepath)
    subset_size = math.ceil(len(data) / num_subset)
    for i in range(num_subset):
        temp = data[i*subset_size:(i+1)*subset_size]
        save_as_json(temp, f'./data/cleaned_data/weibo_contents_subset_{i}.json')


def ner_pipeline_by_subset(datatype, model_name, subset=None):
    set_workspace()
    ner = NER(model_name=model_name)
    logger = Logger(f'ner_{datatype}_{model_name}_subset_{subset}').logger
    print('log files created.')
    logger.info(f'----------------Task NER with model {model_name} subset {subset} start!')
    task_start = time.time()
    logger.info(f'NER model {model_name} loaded.')

    data_contents = open_json(f'./data/cleaned_data/{datatype}_contents_subset_{subset}.json')
    logger.info(f'Data {datatype}_contents_subset_{subset}.json loaded.')
    batch_size = 200
    num_batch = math.ceil(len(data_contents) / batch_size)
    logger.info(f"NER subset {subset}, Number of Batches: {num_batch}--------------------")
    i = 1
    while i <= num_batch:
        try:
            logger.info(f"-------NER batch {i} start.")
            start = time.time()
            # batch = [data for data in data_contents[i * 100:(i + 1) * 100] if isinstance(data, str)]
            batch = data_contents[(i-1) * batch_size:i * batch_size]
            logger.info(f'Batch size {len(batch)}')
            ner_contents = list(map(lambda item: (item['weibo_id'], ner.get_ner_results(item['content'])), batch))
            logger.info(f'ner results len {len(ner_contents)}')
        except Exception as e:
            logger.info(e)
        # ner_content_formatted = ner.formatting_ner_results(ner_contents)
        save_as_json(ner_contents, f'./data/results/ner_3/ner_{datatype}_subset_{subset}_batch_{i}_content.json')
        logger.info(f'-------NER subset {subset} batch {i} data ner_{datatype}_content_formatted.json saved.')
        end = time.time()
        logger.info(f"-------NER batch {i} has completed in {end-start} s.")
        logger.info(f"-------NER subset {subset} has completed {round(i*1.0/num_batch, 4)}------")

        i += 1

    logger.info(f'----------------NER subset {subset} all batch done.')
    # print('Subsets Done.')


def gather_ner_by_subset(subset_no=None):
    if subset_no is None:
        subset = list(filter(lambda x: f'subset_' in x, ner_results_files))
    else:
        subset = list(filter(lambda x: f'subset_{subset_no}_' in x, ner_results_files))
    ner_results = []
    for batch in subset:
        ner_results.extend(open_json(f'{result_path}/{batch}'))
        ner_results_clean = list(filter(lambda ner: ner[0] in weibo_id_base, ner_results))
    save_as_json(ner_results_clean, f'./data/results/ner_3/ner_results_subset_{subset_no}.json')

    print(f'ner subset {subset_no} gathered and saved.')


if __name__ == '__main__':

    set_workspace()
    # mkdir('./data/results/ner_3')
    # # step 1
    # # prepare the text
    # # weibo content
    # model_name = 'clue-ner'
    # p = Pool(19)
    # # get_content_text('weibo')
    # filepath = f'./data/cleaned_data/weibo_contents_1124.json'
    # get_subset(filepath, 19)
    # for i in range(19):
    #     p.apply_async(ner_pipeline_by_subset, ('weibo', model_name), {'subset': i, })
    # p.close()
    # p.join()

    # step 2  gather results by subset
    result_path = './data/results/ner_3'

    global ner_results_files, weibo_id_base
    ner_results_files = os.listdir(result_path)
    weibo_id_base = set(open_json('./data/cleaned_data/weibo_id_base.json'))

    # for i in range(19):
    #     gather_ner_by_subset(i)
    gather_ner_by_subset()





