# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import traceback
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from data_handler.streaming import StreamReaderForSpeedy
from utility.utils import MODEL_CLASSES


class DataLoaderTrainForSpeedyRec(IterableDataset):
    '''
    DataLoader used for training with producer-consumer architecture.
    - Dynamic Batching
    - Generate batch for two-stage encoding
    '''
    def __init__(self,
                 args,
                 data_files,
                 cache_state,
                 end,
                 local_rank,
                 world_size,
                 news_features,
                 news_index,
                 enable_prefetch=True,
                 enable_prefetch_stream=True,
                 global_step=0,
                 add_pad_news=False):
        '''
        Args:
            args: parameters
            data_files(shared list): the paths of train data, storaged in a shared list
            news_idx_incache(shared dict): {news_id:(index in cache, encoded step)}
            end(shared bool value): If it is True, stop all data processes
            local_rank(int): The rank of current process
            world_size(int): The number of processes
            news_features(ndarray):{news_id:(segments_ids, segments_mask, key_position, key_frequence, elements)}
        '''
        self.args = args
        self.beta_for_cache = args.beta_for_cache
        self.cache_state = cache_state
        self.end = end
        self.local_rank = local_rank
        self.world_size = world_size
        self.enable_prefetch = enable_prefetch
        self.enable_prefetch_stream = enable_prefetch_stream
        self.global_step = global_step

        self.local_end = False
        self.global_end = end
        self.global_end.value = False

        self.news_features = news_features
        self.news_index = news_index

        self.data_files = data_files
        self.sub_files = [x for x in data_files]
        self.batch_size = args.batch_size*args.world_size

        self.cache_state = [-args.max_step_in_cache - 1000]*len(news_features)
        # for nid, inx in news_index.items():
        #     self.cache_state[inx] = -args.max_step_in_cache - 1000
        self.add_pad_news = add_pad_news

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        # self.end.value = False
        if self.enable_prefetch:
            self.start_async()
        else:
            self.outputs = self.dynamic_batch().__iter__()
        return self

    def start_async(self):
        logging.info('start async...')
        self.aval_count = 0
        # self.end.value = False
        self.outputs = Queue(5)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)


    def __next__(self):
        dist.barrier()
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value=True
                break
        dist.barrier()
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

    def _produce(self):
        try:
            for address_cache, update_cache, start_inx, end_inx, batch in self.dynamic_batch():
                self.global_step += 1
                self.update_use_cache()

                self.outputs.put((address_cache, update_cache, start_inx, end_inx, batch))
                self.aval_count += 1
            self.local_end = True
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
            self.pool.shutdown(wait=False)
            raise
            # self.pool.shutdown(wait=False)
            # raise

    def dynamic_batch(self):
        '''
        Each training case will be routed to a bucket based on its max sequence length.
        The buckets will be checked before each insert-in: that whether it is filled.
        Once a bucket is filled, the filled transactions will be generated as a mini-batch and appended to the mini-batch queue.
        '''

        self.use_cache = False
        self.timer = 0
        if self.args.enable_gpu:
            torch.cuda.set_device(self.local_rank)

        self.sampler_batch = StreamReaderForSpeedy(
            file=self.sub_files,
            batch_size=self.batch_size
            )

        for batch in self.sampler_batch:
            news_set, uid_click_docs, uid_sample_news = self._process(batch)
            cache_set, encode_set = self.split_news_set(news_set, self.use_cache)
            assert 0 not in encode_set

            address_cache, update_cache, start_inx, end_inx, batch = self.gen_batch_for_two_stage(encode_set,
                                                                              cache_set,
                                                                              uid_click_docs,
                                                                              uid_sample_news)
            yield address_cache, update_cache, start_inx, end_inx, batch

        # self.end.value = True
        self.local_end = True

    def drop_encoder_prob(self, step):
        return self.args.max_hit_ratio - math.exp(max(-step*self.beta_for_cache,-1000))

    def update_use_cache(self):
        if random.random() < self.drop_encoder_prob(self.global_step):
            self.use_cache = True
        else:
            self.use_cache = False

    def split_news_set(self,news_set,use_cache):
        '''
        For each news article, the dataloader will check the cache in the first place:
        if there is a copy of news embedding in cache, it will outputs the index of news in the cache
        otherwise, it will outputs the features of this news as imputs to news encoder.
        '''
        # stime = time.time()
        if use_cache:
            cache_set = []
            encode_set = []
            for n in news_set:
                if n == 0:
                    continue
                if self.global_step - self.cache_state[n] <= self.args.max_step_in_cache:
                    cache_set.append(n)
                else:
                    encode_set.append(n)

            return cache_set, encode_set
        else:
            news_set.discard(0)

            return set(), news_set


    def gen_batch_for_two_stage(self,
                                encode_news_set,
                                cache_news_set,
                                uid_click_docs,
                                uid_sample_news):
        '''
        Once a mini-batch is presented, it will gather all of the news articles from different users.
        '''
        batch_news_index = {0 : 0}
        idx = 1

        if len(cache_news_set)==0:
            address_cache = None
        else:
            address_cache = []
            for n in cache_news_set:
                address_cache.append(n)
                batch_news_index[n] = idx
                idx += 1
            address_cache = np.array(address_cache)

        update_cache = []
        encode_news_set = list(encode_news_set)
        for n in encode_news_set:
            batch_news_index[n] = idx
            idx += 1
            self.cache_state[n] = self.global_step
            update_cache.append(n)

        if self.add_pad_news:
            add_pad_num = math.ceil(len(encode_news_set)/self.world_size)*self.world_size - len(encode_news_set)
            encode_news_set = encode_news_set + [0]*add_pad_num

        encode_num = len(encode_news_set)//self.world_size
        input_ids = []
        start_inx = self.local_rank*encode_num
        end_inx = (self.local_rank+1)*encode_num
        if self.local_rank == self.world_size-1:
            end_inx = len(encode_news_set)
        for n in encode_news_set[start_inx:end_inx]:
            input_ids.append(self.news_features[n])

        hist_sequence = []
        hist_sequence_mask = []
        candidate_inx = []

        # max_hist_len = self.args.user_log_length
        max_hist_len = max([len(x) for x in uid_click_docs])+1 #add a learnable pad_doc

        label_batch = []

        for click_docs, sample_news in zip(uid_click_docs, uid_sample_news):
            history = self.trans_to_batchindex(click_docs, batch_news_index)
            history, mask = self.pad_to_fix_len(history, max_hist_len)
            hist_sequence.append(history)
            hist_sequence_mask.append(mask)

            pos_neg = self.trans_to_batchindex(sample_news, batch_news_index)
            candidate_inx.append(pos_neg)
            label_batch.append(0)

        if self.args.enable_gpu:
            input_ids = torch.LongTensor(input_ids).cuda()
            hist_sequence = torch.LongTensor(hist_sequence).cuda()
            hist_sequence_mask = torch.FloatTensor(hist_sequence_mask).cuda()
            candidate_inx = torch.LongTensor(candidate_inx).cuda()
            label_batch = torch.LongTensor(label_batch).cuda()

        else:
            input_ids = torch.LongTensor(input_ids)
            hist_sequence = torch.LongTensor(hist_sequence)
            hist_sequence_mask = torch.FloatTensor(hist_sequence_mask)
            candidate_inx = torch.LongTensor(candidate_inx)
            label_batch = torch.LongTensor(label_batch).cuda()

        return address_cache, \
               np.array(update_cache),\
               start_inx, end_inx, \
               (input_ids, hist_sequence, hist_sequence_mask, candidate_inx, label_batch)

    def _process(self, batch):
        random.seed(self.global_step)
        batch = [x.decode(encoding="utf-8").split("\t") for x in batch]
        news_set, behavior_set, uid_click_docs, uid_sample_news = [], [], [], []
        for line in batch:
            click_docs = [i for i in line[2].split()]
            poss = line[3]

            click_docs = self.trans_to_nindex(click_docs)[-self.args.user_log_length:]
            sess_neg = [i for i in line[4].split()]
            poss = self.trans_to_nindex([poss])
            sess_neg = self.trans_to_nindex(sess_neg)

            if len(sess_neg) > 0:
                neg_index = self.news_sample(list(range(len(sess_neg))),
                                        self.args.npratio)
                sam_negs = [sess_neg[i] for i in neg_index]
            else:
                sam_negs = [0] * self.args.npratio

            sample_news = poss + sam_negs

            news_set.extend(sample_news+click_docs)

            uid_click_docs.append(click_docs)
            uid_sample_news.append(sample_news)

        news_set = set(news_set)
        return news_set, uid_click_docs, uid_sample_news


    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def trans_to_batchindex(self, nids, news_index):
        return [news_index[i] for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_value=0, padding_front=True):
        if padding_front:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        else:
            pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length-len(x))
        return pad_x,mask

    def news_sample(self, news, ratio):
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        else:
            return random.sample(news, ratio)

