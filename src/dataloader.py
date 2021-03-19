# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import traceback
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from .streaming import StreamSamplerTest, StreamSamplerTrain
import logging
import math

class DataLoaderTrain(IterableDataset):
    '''
    DataLoader used for training with producer-consumer architecture.
    - Dynamic Batching
    - Generate batch for two-stage encoding
    '''
    def __init__(self,
                 args,
                 data_files,
                 news_idx_incache,
                 prefetch_step,
                 end,
                 local_rank,
                 world_size,
                 news_features,
                 enable_prefetch=True,
                 enable_prefetch_stream=False,
                 global_step=0):
        '''
        Args:
            args: parameters
            data_files(shared list): the paths of train data, storaged in a shared list
            news_idx_incache(shared dict): {news_id:(index in cache, encoded step)}
            prefetch_step(shared list): sync the dataloaders
            end(shared bool value): If it is True, stop all data processes
            local_rank(int): The rank of current process
            world_size(int): The number of processes
            news_features(dict):{news_id:(segments_ids, segments_mask, key_position, key_frequence, elements)}
        '''
        self.args = args
        self.beta_for_cache = args.beta_for_cache
        self.data_files = data_files
        self.news_idx_incache = news_idx_incache
        self.prefetch_step = prefetch_step
        self.end = end
        self.local_rank = local_rank
        self.world_size = world_size
        self.news_features = news_features
        self.enable_prefetch = enable_prefetch
        self.enable_prefetch_stream = enable_prefetch_stream
        self.global_step = global_step

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        if self.enable_prefetch:
            self.start_async()
        else:
            self.outputs = self.dynamic_batch().__iter__()
        return self

    def start_async(self):
        logging.info('start async...')
        self.aval_count = 0
        self.end.value = False
        self.prefetch_step[self.local_rank] = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def __next__(self):
        if self.enable_prefetch:
            if self.end.value and self.aval_count == 0:
                raise StopIteration
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
            return next_batch
        else:
            return self.outputs.__next__()

    def _produce(self):
        for address_cache,update_cache,batch in self.dynamic_batch():
            self.outputs.put((address_cache, update_cache, batch))
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def dynamic_batch(self):
        '''
        Each training case will be routed to a bucket based on its max sequence length.
        The buckets will be checked before each insert-in: that whether it is filled.
        Once a bucket is filled, the filled transactions will be generated as a mini-batch and appended to the mini-batch queue.
        '''
        # Buffer Buckets
        logging.info('init bucket')
        blocks = [[]for x in range(self.args.bucket_num)]   #hitory_id, neg_id
        block_encode_set = [set() for x in range(self.args.bucket_num)]
        block_cache_set = [set() for x in range(self.args.bucket_num)]
        block_max_length = [0 for x in range(self.args.bucket_num)]
        block_space = [(self.args.seg_length // self.args.bucket_num) * i for i in range(self.args.bucket_num)]

        self.use_cache = False

        if self.args.enable_gpu:
            torch.cuda.set_device(self.local_rank)

        self.sampler = StreamSamplerTrain(data_files=self.data_files)
        if self.enable_prefetch_stream:
            self.sampler_batch = self.sampler
        else:
            self.sampler_batch = self.sampler._generate_batch()

        for one_user in self.sampler_batch:
            news_set, history, negs = self._process(one_user)
            cache_set, encode_set = self.split_news_set(news_set, self.use_cache)
            max_len = 0
            if len(encode_set) > 0:
                max_len = max([len(self.news_features[nid][0][0]) if nid in self.news_features else 0 for nid in encode_set])

            for i in range(self.args.bucket_num-1,-1,-1):
                if max_len > block_space[i]:
                    if (max(block_max_length[i],max_len)+self.args.bus_num)*len(block_encode_set[i] | encode_set)*self.args.seg_num > self.args.batch_size:
                        if len(block_encode_set[i]) == 0:
                            break
                        address_cache,update_cache,batch = self.gen_batch_for_two_stage(block_encode_set[i],block_cache_set[i],blocks[i],block_max_length[i],self.global_step)

                        self.prefetch_step[self.local_rank] += 1
                        self.synchronization()
                        if self.end.value:
                            break

                        self.global_step += 1
                        yield address_cache,update_cache,batch

                        block_encode_set[i] = set();block_cache_set[i] = set();blocks[i]=[];block_max_length[i]=0
                        self.update_use_cache()

                    block_max_length[i] = max(block_max_length[i],max_len)
                    block_encode_set[i] = block_encode_set[i] | encode_set
                    block_cache_set[i] = block_cache_set[i] | cache_set
                    blocks[i].append((history,negs))
                    break
        self.end.value = True

    def synchronization(self):
        while sum(self.prefetch_step) != self.prefetch_step[self.local_rank] * self.world_size:
            if self.end.value: break

    def drop_encoder_prob(self, step):
        return 1 - math.exp(-step*self.beta_for_cache)

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
        if use_cache:
            cache_set = set()
            encode_set = set()
            for n in news_set:
                if n == 'MISS':
                    continue
                if self.global_step - self.news_idx_incache[n][1] <= self.args.max_step_in_cache:
                    cache_set.add(n)
                else:
                    encode_set.add(n)

            return cache_set,encode_set
        else:
            news_set.discard('MISS')
            return set(), news_set


    def gen_batch_for_two_stage(self,encode_set,cache_set,data,max_len,global_step):
        '''
        Once a mini-batch is presented, it will gather all of the news articles from different users.
        '''
        news_index = {'MISS': 0}
        idx = 1

        if len(cache_set)==0:
            address_cache = None
        else:
            address_cache = []
            for n in cache_set:
                address_cache.append(self.news_idx_incache[n][0])
                news_index[n] = idx
                idx += 1
            address_cache = np.array(address_cache)

        update_cache = []
        segments = []
        token_masks = []
        seg_masks = []
        key_position = []
        fre_ids = []
        elements = []
        for n in encode_set:
            news_index[n] = idx
            idx += 1

            tokens,s_mask,positions,fre_cnt, elem = self.news_features[n]
            for i in range(self.args.seg_num):
                text,mask = self.pad_to_fix_len(tokens[i],max_len,padding_front=False)
                segments.append(text)
                token_masks.append(mask)
                if self.args.content_refinement:
                    position,_ = self.pad_to_fix_len(positions[i],max_len,padding_front=False)
                    key_position.append(position)
                    fre,_ = self.pad_to_fix_len(fre_cnt[i],max_len,padding_front=False)
                    fre = [min(x,self.args.max_keyword_freq-1) for x in fre]
                    fre_ids.append(fre)

            seg_masks.append(s_mask[:self.args.seg_num])
            elements.append(elem)

            # update cache
            update_cache.append(self.news_idx_incache[n][0])
            self.news_idx_incache[n] = [self.news_idx_incache[n][0],global_step]

        batch_hist = []
        batch_negs = []
        batch_mask = []
        max_hist_len = max([len(x[0]) for x in data])
        for history,negs in data:
            history = self.trans_to_nindex(history,news_index)
            history,mask = self.pad_to_fix_len(history,max_hist_len)
            batch_hist.append(history)
            batch_mask.append(mask)

            temp_negs = [self.trans_to_nindex(n,news_index) for n in negs]
            temp_negs = self.pad_to_fix_len_neg(temp_negs,max_hist_len-1)
            batch_negs.append(temp_negs)

        if self.args.enable_gpu:
            segments = torch.LongTensor(segments).cuda()
            token_masks = torch.FloatTensor(token_masks).cuda()
            seg_masks = torch.FloatTensor(seg_masks).cuda()
            if self.args.content_refinement:
                key_position = torch.LongTensor(key_position).cuda()
                fre_ids = torch.LongTensor(fre_ids).cuda()
            else:
                key_position, fre_ids = None,None
            elements = torch.LongTensor(elements).cuda()
            batch_hist = torch.LongTensor(batch_hist).cuda()
            batch_negs = torch.LongTensor(batch_negs).cuda()
            batch_mask = torch.FloatTensor(batch_mask).cuda()
        else:
            segments = torch.LongTensor(segments)
            token_masks = torch.FloatTensor(token_masks)
            seg_masks = torch.FloatTensor(seg_masks)
            if self.args.content_refinement:
                key_position = torch.LongTensor(key_position)
                fre_ids = torch.LongTensor(fre_ids)
            else:
                key_position, fre_ids = None, None
            elements = torch.LongTensor(elements)
            batch_hist = torch.LongTensor(batch_hist)
            batch_negs = torch.LongTensor(batch_negs)
            batch_mask = torch.FloatTensor(batch_mask)

        return address_cache,np.array(update_cache),(segments,token_masks,seg_masks,key_position,fre_ids, elements,batch_hist,batch_mask,batch_negs)

    def trans_to_nindex(self, nids,news_index):
        return [news_index[i] for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_value=0, padding_front=True):
        if padding_front:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        else:
            pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length-len(x))
        return pad_x,mask

    def pad_to_fix_len_neg(self, x, fix_length, padding_value=0,padding_front=True):
        if padding_front:
            pad_x = x[-fix_length:] + [[padding_value] * self.args.npratio] * (fix_length - len(x))
        else:
            pad_x = x[:fix_length] + [[padding_value] * self.args.npratio] * (fix_length - len(x))
        return pad_x

    def _process(self, line):
        clicked = []
        negnews = []
        u_set = []

        uid, sessions = line.strip().split('\t')
        for sess in sessions.split('|'):
            pos, neg = sess.split('&')
            pos = [p if p in self.news_features else 'MISS' for p in pos.split(';')]
            clicked.extend(pos)

            neg = neg.split(';')
            for p in pos:
                if len(neg) < self.args.npratio:
                    neg = neg*(int(self.args.npratio/len(neg))+1)
                sample_neg = [n if n in self.news_features else 'MISS' for n in random.sample(neg, self.args.npratio)]
                negnews.append(sample_neg)

        clicked = clicked[-self.args.user_log_length:]
        negnews = negnews[-(self.args.user_log_length-1):]

        for p in clicked:
            u_set.append(p)
        for ns in negnews:
            u_set.extend(ns)

        u_set = set(u_set)
        return u_set,clicked,negnews




class DataLoaderTest():
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring,
                 news_bias_scoring=None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.test_batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def pad_to_fix_len_neg(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [[padding_value] * self.npratio] * (fix_length - len(x))
        return pad_x

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []

        for line in batch:
            uid, click_docs, pos, neg = line.decode(encoding="utf-8").split('\t')

            click_docs = click_docs.split(';')
            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                                       self.user_log_length)
            user_feature = self.news_scoring[click_docs]

            sess_pos = pos.split(';')
            sess_neg = neg.split(';')
            sess_pos = self.trans_to_nindex(sess_pos)
            sess_neg = self.trans_to_nindex(sess_neg)

            sample_news = sess_pos + sess_neg
            labels = [1] * len(sess_pos) + [0] * len(sess_neg)

            news_feature = self.news_scoring[sample_news]

            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            label_batch.append(np.array(labels))

        if self.enable_gpu:
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()

        else:
            user_feature_batch = torch.FloatTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, label_batch

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None
