# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import logging
import random
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np

from LanguageModels.SpeedyModel import SpeedyModelForRec
from LanguageModels.configuration_tnlrv3 import TuringNLRv3Config

from .utils import (setuplogging, init_process, cleanup_process, warmup_linear, init_config, dump_args, get_device,
                    get_barrier, only_on_main_process, check_args_environment)
from .streaming import get_files
from .dataloader import DataLoaderTrain
from .preprocess import read_news, check_preprocess_result
from .speedyfeed import SpeedyFeed


def ddp_train(args):
    '''
    Distributed training
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    setuplogging()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args = check_args_environment(args)
    logging.info('-----------start train------------')
    if args.world_size > 1:
        cache = np.zeros((args.cache_num, args.news_dim))
        global_cache = mp.Manager().list([cache])
        news_idx_incache = mp.Manager().dict()
        global_prefetch_step = mp.Manager().list([0] * args.world_size)
        data_files = mp.Manager().list([])
        end = mp.Manager().Value('b', False)
        mp.spawn(train,
                 args=(args, global_cache, news_idx_incache, global_prefetch_step, end, data_files),
                 nprocs=args.world_size,
                 join=True)
    else:
        cache = [np.zeros((args.cache_num,
                           args.news_dim))]
        news_idx_incache = {}
        prefetch_step = [0]
        data_files = []
        end = mp.Manager().Value('b', False)
        train(0, args, cache, news_idx_incache, prefetch_step, end, data_files, dist_training=False)

def train(local_rank,
          args,
          cache,
          news_idx_incache,
          prefetch_step,
          end,
          data_files, 
          dist_training=True):
    '''
    Args:
        local_rank(int): the rank of current process
        args: parameters
        cache(shared list): global shared cache, the vec will be storaged in it as numpy.array
        news_idx_incache(shared dict): {news_id:(index in cache, encoded step)}
        prefetch_step(shared list): sync the dataloaders
        end(shared bool): If it is True, stop all data processes
        data_files(shared list): the paths of train data, storaged in a shared list
    '''
    setuplogging()
    cache = cache[0]

    if dist_training:
        init_process(local_rank, args.world_size)
    device = get_device()
    barrier = get_barrier(dist_training)

    logging.info('loading model: {}'.format(args.bert_model))

    args, config = init_config(args, TuringNLRv3Config)
    bert_model = SpeedyModelForRec.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config)

    if args.freeze_bert:
        logging.info('Freeze the parameters of {}'.format(args.bert_model))
        for param in bert_model.parameters():
            param.requires_grad = False

        # choose which block trainable
        for index, layer in enumerate(bert_model.bert.encoder.layer):
            if index in args.finetune_blocks:
                logging.info(f"finetune block {index}")
                for param in layer.parameters():
                    param.requires_grad = True

    root_data_dir = os.path.join(args.root_data_dir,'traindata')
    with only_on_main_process(local_rank, barrier) as need:
        if need:
            check_preprocess_result(args,root_data_dir)
            logging.info('finish the preprocess of docfeatures')

    news_features, category_dict, subcategory_dict = read_news(args,root_data_dir)
    logging.info('news_num:{}'.format(len(news_features)))

    #init the news_idx_incache and data_paths
    assert args.cache_num >= len(news_features)
    
    with only_on_main_process(local_rank, barrier) as need:
        if need:
            for idx, news in enumerate(news_features.keys()):
                news_idx_incache[news] = [idx, -args.max_step_in_cache]
                
            data_paths = get_files(dirname=os.path.join(args.root_data_dir,'traindata'),
                                filename_pat=args.filename_pat)
            data_paths.sort()
            dump_args(args)

    model = SpeedyFeed(args, bert_model, len(category_dict),
                       len(subcategory_dict))
    model = model.to(device)
    if dist_training:
        ddp_model = DDP(model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True)
    else:
        ddp_model = model

    rest_param = filter(
        lambda x: id(x) not in list(map(id, bert_model.parameters())),
        ddp_model.parameters())
    optimizer = optim.Adam([{
        'params': bert_model.parameters(),
        'lr': args.pretrain_lr * warmup_linear(args, 1)
    }, {
        'params': rest_param,
        'lr': args.lr * warmup_linear(args, 1)
    }])

    logging.info('Training...')
    start_time = time.time()
    global_step = 0
    for ep in range(args.epochs):
        with only_on_main_process(local_rank, barrier) as need:
            # data_files.clear()
            if need:
                while len(data_files) > 0:
                    data_files.pop()
                data_files.extend(data_paths)
                random.seed(ep)
                random.shuffle(data_files)

        dataloader = DataLoaderTrain(
            args=args,
            data_files=data_files,
            news_idx_incache=news_idx_incache,
            prefetch_step=prefetch_step,
            end=end,
            local_rank=local_rank,
            world_size=args.world_size,
            news_features=news_features,
            enable_prefetch=args.enable_prefetch,
            enable_prefetch_stream=args.enable_prefetch_stream)

        loss = 0.0
        usernum = 0
        for cnt, batch in tqdm(enumerate(dataloader)):
            address_cache, update_cache, batch = batch
            usernum += batch[-3].size(0)
            global_step += 1

            if args.enable_gpu:
                segments, token_masks, seg_masks, key_position, fre_cnt, elements, batch_hist, batch_mask, batch_negs = (
                    x.cuda(non_blocking=True) if x is not None else x
                    for x in batch)
            else:
                segments, token_masks, seg_masks, key_position, fre_cnt, elements, batch_hist, batch_mask, batch_negs = batch

            #Get news vecs from cache.
            if address_cache is not None:
                cache_vec = torch.FloatTensor(
                    cache[address_cache]).cuda(non_blocking=True)
            else:
                cache_vec = None

            bz_loss,encode_vecs = ddp_model(segments,token_masks, seg_masks, elements, cache_vec,
                                            batch_hist, batch_mask, batch_negs,
                                            key_position, fre_cnt)
            loss += bz_loss.item()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            #update the cache
            if args.drop_encoder_ratio > 0:
                encode_vecs = encode_vecs.detach().cpu().numpy()
                cache[update_cache] = encode_vecs

            optimizer.param_groups[0]['lr'] = args.pretrain_lr*warmup_linear(args,global_step+1)  #* world_size
            optimizer.param_groups[1]['lr'] = args.lr*warmup_linear(args,global_step+1)   #* world_size

            if global_step % args.log_steps == 0:
                logging.info(
                    '[{}] cost_time:{} step:{}, usernum: {}, train_loss: {:.5f}, lr:{}, pretrain_lr:{}'.format(
                        local_rank, time.time() - start_time, global_step, usernum, loss / args.log_steps,
                        optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))
                loss = 0.0

            # save model minibatch
            if local_rank == 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep + 1}-{global_step}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'category_dict': category_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

        logging.info('epoch:{}, usernum:{}, time:{}'.format(ep + 1, usernum, time.time() - start_time))
        # save model after an epoch
        if local_rank == 0:
            ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

    if dist_training:
        cleanup_process()
