# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import math
from multiprocessing import cpu_count
from multiprocessing import Pool
from src.utils import word_tokenize
from nltk.corpus import stopwords
from src.parameters import parse_args
import os

class BM25(object):
    '''
    The implementation of BM25
    '''
    def __init__(self, corpus_entity, entity_list, k2=2, epsilon=0.25):
        self.k2 = k2
        self.epsilon = epsilon
        self.entity_list = entity_list
        self.corpus_size = len(corpus_entity)
        self.idf = {entity:0 for entity in entity_list}
        self.freq = {entity:0 for entity in entity_list}

        self.cal_freq(corpus_entity)
        self.cal_idf()

    def cal_freq(self, corpus_entity):
        for news_entities in corpus_entity:
            for entity in news_entities:
                self.freq[entity] += 1


    def cal_idf(self):
        idf_sum = 0
        idf_pos_cnt = 0
        neg_idfs = []
        for entity in self.freq:
            idf = math.log(self.corpus_size - self.freq[entity] + 0.5) - math.log(self.freq[entity] + 0.5)
            if idf > 0:
                self.idf[entity] = idf
                idf_sum += idf
                idf_pos_cnt += 1
            else:
                neg_idfs.append(entity)
        avg_idf = idf_sum / idf_pos_cnt

        for entity in neg_idfs:
            self.idf[entity] = avg_idf * self.epsilon

    def scoring(self, news_entity):
        '''
        Calculate the score of words in a news
        Args:
            news_entity(dict): {word:cnt}
        Returns:
            res(list):[(word,cnt,score)]
        '''
        scores = []
        res = []
        for e, qf in news_entity.items():
            score = self.idf[e] * qf * (self.k2 + 1) / (qf + self.k2)
            scores.append(score)
            res.append((e,qf,score))

        res = sorted(res, key=lambda s: s[-1], reverse=True)
        return res


def get_words(text):
    text = text.lower()
    temp_words = word_tokenize(text)
    temp_text_dict = {}
    temp_word_list = set()
    for w in temp_words:
        if w not in temp_text_dict:
            temp_text_dict[w] = 1
            temp_word_list.add(w)
        else:
            temp_text_dict[w] += 1
    return temp_text_dict,temp_word_list


def mul_preprocess(local_rank,world_size,data_path):
    news_dict = {}
    title_corpus = []
    title_word_list = set()
    abs_corpus = []
    abs_word_list = set()
    body_corpus = []
    body_word_list = set()

    f = open(data_path, 'r', encoding='utf-8')
    for i, line in enumerate(f):
        if i % world_size != local_rank: continue

        splited = line.strip('\n').split('\t')
        doc_id, category, subcategory, title, abstract, body = splited

        temp_title_dict, temp_title_word = get_words(title)
        title_corpus.append(list(temp_title_dict.keys()))
        title_word_list = title_word_list | temp_title_word

        temp_abs_dict, temp_abs_word = get_words(abstract)
        abs_corpus.append(list(temp_abs_dict.keys()))
        abs_word_list = abs_word_list | temp_abs_word

        temp_body_dict, temp_body_word = get_words(body)
        body_corpus.append(list(temp_body_dict.keys()))
        body_word_list = body_word_list | temp_body_word

        news_dict[doc_id] = (temp_title_dict,temp_abs_dict,temp_body_dict)

    return news_dict,\
           title_corpus, title_word_list, \
           abs_corpus, abs_word_list,\
           body_corpus, body_word_list


def content_refinement(args,world_size):
    k2 = args.k2_for_BM25
    data_path = os.path.join(args.root_data_dir, 'DocFeatures.tsv')
    topk = 100

    pool = Pool(processes=world_size)
    results = []
    for i in range(world_size):
        result = pool.apply_async(mul_preprocess, args=(i, world_size, data_path))
        results.append(result)
    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('All subprocesses done.')

    news_text = {}
    title_corpus, title_word_list, abs_corpus, abs_word_list, body_corpus, body_word_list = [], set(), [], set(), [], set()
    for result in results:
        news_dict, p_title_corpus, p_title_word_list, p_abs_corpus, p_abs_word_list, p_body_corpus, p_body_word_list = result.get()

        title_corpus.extend(p_title_corpus)
        title_word_list = title_word_list | p_title_word_list

        abs_corpus.extend(p_abs_corpus)
        abs_word_list = abs_word_list | p_abs_word_list

        body_corpus.extend(p_body_corpus)
        body_word_list = body_word_list | p_body_word_list

        for k, v in news_dict.items():
            news_text[k] = v
    print('news num: {}'.format(len(news_text)))

    title_word_list = list(title_word_list)
    abs_word_list = list(abs_word_list)
    body_word_list = list(body_word_list)
    title_bm25 = BM25(title_corpus, title_word_list, k2)
    abs_bm25 = BM25(abs_corpus, abs_word_list, k2)
    body_bm25 = BM25(body_corpus, body_word_list, k2)

    stpwords = stopwords.words('english')
    punctuation = ['!', ',', '.', '?', '\\', '-', '|']
    stpwords.extend(punctuation)

    news_keywords = {}
    for doc_id, text in news_text.items():
        titel_key = title_bm25.scoring(text[0])
        titel_key = [x for x in titel_key if x[0] not in stpwords][:topk]
        abs_key = abs_bm25.scoring(text[1])
        abs_key = [x for x in abs_key if x[0] not in stpwords][:topk]
        body_key = body_bm25.scoring(text[2])
        body_key = [x for x in body_key if x[0] not in stpwords][:topk]

        news_keywords[doc_id] = (titel_key, abs_key, body_key)
        if len(news_keywords) % 10000 == 0:
            print(f'Have processed {len(news_keywords)} news')

    data_path = os.path.join(args.root_data_dir, 'refinement_k2={}.pkl'.format(k2))
    with open(data_path, 'wb') as f:
        pickle.dump(news_keywords, f)

#
if __name__ == '__main__':
    args = parse_args()

    # k2 = args.k2_for_BM25
#     data_path = os.path.join(args.root_data_dir,'DocFeatures.tsv')
#     world_size = args.num_worker_preprocess
#     topk = 100
#
#     pool = Pool(processes=world_size)
#     results = []
#     for i in range(world_size):
#         result = pool.apply_async(mul_preprocess, args=(i,world_size,data_path))
#         results.append(result)
#     print('Waiting for all subprocesses done...')
#     pool.close()
#     pool.join()
#     print('All subprocesses done.')
#
#     news_text = {}
#     title_corpus, title_word_list, abs_corpus, abs_word_list, body_corpus, body_word_list = [],set(),[],set(),[],set()
#     for result in results:
#         news_dict, p_title_corpus, p_title_word_list, p_abs_corpus, p_abs_word_list, p_body_corpus, p_body_word_list = result.get()
#
#         title_corpus.extend(p_title_corpus)
#         title_word_list = title_word_list | p_title_word_list
#
#         abs_corpus.extend(p_abs_corpus)
#         abs_word_list = abs_word_list | p_abs_word_list
#
#         body_corpus.extend(p_body_corpus)
#         body_word_list = body_word_list | p_body_word_list
#
#         for k,v in news_dict.items():
#             news_text[k] = v
#     print('news num: {}'.format(len(news_text)))
#
#     title_word_list = list(title_word_list)
#     abs_word_list = list(abs_word_list)
#     body_word_list = list(body_word_list)
#     title_bm25 = BM25(title_corpus, title_word_list, k2)
#     abs_bm25 = BM25(abs_corpus, abs_word_list, k2)
#     body_bm25 = BM25(body_corpus, body_word_list, k2)
#
#     stpwords = stopwords.words('english')
#     punctuation = ['!', ',', '.', '?','\\','-','|']
#     stpwords.extend(punctuation)
#
#     news_keywords = {}
#     for doc_id,text in news_text.items():
#         titel_key = title_bm25.scoring(text[0])
#         titel_key = [x for x in titel_key if x[0] not in stpwords][:topk]
#         abs_key = abs_bm25.scoring(text[1])
#         abs_key = [x for x in abs_key if x[0] not in stpwords][:topk]
#         body_key = body_bm25.scoring(text[2])
#         body_key = [x for x in body_key if x[0] not in stpwords][:topk]
#
#         news_keywords[doc_id] = (titel_key,abs_key,body_key)
#         if len(news_keywords)%10000 == 0:
#             print(f'Have processed {len(news_keywords)} news')
#
#     data_path = '../example_data/refinement.pkl'
#     with open(data_path, 'wb') as f:
#         pickle.dump(news_keywords, f)
#

