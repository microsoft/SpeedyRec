# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pickle
from tqdm import tqdm
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader

from utility.utils import MODEL_CLASSES


class NewsInfo:
    def __init__(self, args, mode='train', category_dict=None, subcategory_dict=None):
        self.args = args
        self.mode = mode
        self.news = {}
        self.news_index = {}

        if self.mode != 'train':
            self.category_dict, self.subcategory_dict \
                 = category_dict, subcategory_dict
        else:
            self.category_dict, self.subcategory_dict = {}, {}

        _, _, tokenizer_class = MODEL_CLASSES[args.pretreained_model]
        self.tokenizers = tokenizer_class.from_pretrained(self.args.pretrained_model_path, do_lower_case=True)

    def update_dict(self, dict, key, value=None):
        if key not in dict:
            if value is None:
                dict[key] = len(dict) + 1
            else:
                dict[key] = value
        return dict

    def sent_tokenize(self, sent, max_len):
        assert isinstance(sent, str)
        sent_split = self.tokenizers(sent, max_length=max_len, pad_to_max_length=True, truncation=True)
        return sent_split

    def _parse_news_attrs(self, attr_raw_values):
        parser = {
            'title': (self.sent_tokenize, [], {"max_len":self.args.num_words_title}),
            'body': (self.sent_tokenize, [], {"max_len": self.args.num_words_body}),
            'abstract': (self.sent_tokenize, [], {"max_len": self.args.num_words_abstract}),
            'category': (lambda x: x, None, {}),
            'subcategory': (lambda x: x, None, {}),
        }

        news_attrs = [
            self._parse_news_attr(
            attr_name, parser[attr_name], attr_raw_value
            ) for attr_name, attr_raw_value in 
            zip(
                ['title', 'abstract', 'body', 'category', 'subcategory'],
                attr_raw_values
            )]

        return news_attrs

    def _parse_news_attr(self, attr_name, parser, attr_raw_value):
        parser_func, default_value, kwargs = parser
        if attr_name in self.args.news_attributes:
            return parser_func(attr_raw_value, **kwargs)
        else:
            return default_value

    def parse_line(self, line):
        doc_id, category, subcategory, title, abstract, body = line.strip('\n').split('\t')
        
        title = " ".join([category, subcategory, title])
        title, abstract, body, category, subcategory = self._parse_news_attrs(
            [title, abstract, body, category, subcategory]
        )
        self.update_dict(self.news, doc_id, [title, abstract, body, category, subcategory])
        self.update_dict(self.news_index, doc_id)
        if self.mode == 'train':
            self.update_dict(self.category_dict, category)
            self.update_dict(self.subcategory_dict, subcategory)

    def process_news_file(self, file):
        with open(file, "r") as f:
            for line in tqdm(f):
                self.parse_line(line)
            

def get_doc_input(news, news_index, category_dict,
                  subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract),
                                 dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_attmask = None

    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, subcategory = news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_attmask[doc_index] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0


    return news_title, news_title_attmask, news_abstract, news_abstract_attmask, \
           news_body, news_body_attmask, news_category, news_subcategory



def get_news_feature(args, mode='train', category_dict=None, subcategory_dict=None):
    if mode == 'train':
        news_info = NewsInfo(args, mode)
    else:
        news_info = NewsInfo(args, mode, category_dict, subcategory_dict)

    directory, model_name = os.path.split(args.pretrained_model_path)
    cache_file = f'{args.root_data_dir}/{mode}/{model_name}_{"+".join(args.news_attributes)}_preprocessed_docs.pkl'
    if os.path.exists(cache_file):
        logging.info(f'Load cache from {cache_file}')
        with open(cache_file, 'rb') as f:
            news_info = pickle.load(f)
    else:
        # process news info
        news_info.process_news_file(
            os.path.join(args.root_data_dir,
                         f'{mode}/docs.tsv'))
        with open(cache_file, 'wb') as f:
            pickle.dump(news_info, f)
        logging.info(f"Cached data saved at {cache_file}")

    news_title, news_title_attmask, news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask, news_category, news_subcategory = get_doc_input(
        news_info.news, news_info.news_index, news_info.category_dict,
        news_info.subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_attmask, news_abstract, news_abstract_attmask, \
         news_body, news_body_attmask, news_category, news_subcategory]
        if x is not None], axis=1)
    return news_info, news_combined


def infer_news(model, device, news_combined, batch_size=64):
    class NewsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return self.data.shape[0]

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=batch_size,
                                 num_workers=0)

    news_vecs = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.to(device).long()
            news_vec = model.news_encoder(input_ids).squeeze(dim=1)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_vecs.extend(news_vec)

    news_vecs = np.array(news_vecs)
    logging.info("news scoring num: {}".format(news_vecs.shape[0]))
    return news_vecs

