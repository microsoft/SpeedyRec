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

from LanguageModels.SpeedyModel import SpeedyModelForRec
from LanguageModels.configuration_tnlrv3 import TuringNLRv3Config

# from src.utils import
from .utils import (setuplogging, init_process, cleanup_process, warmup_linear,
                    init_config, dump_args)
from .streaming import get_files
from .dataloader import DataLoaderTrain
from .preprocess import read_news, check_preprocess_result
from .speedyfeed import SpeedyFeed


def train(local_rank, args, cache, news_idx_incache, prefetch_step, end,
          data_files):
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
    os.environ["RANK"] = str(local_rank)

    init_process(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)

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

        # choose which block trainabel
        for index, layer in enumerate(bert_model.bert.encoder.layer):
            if index in args.finetune_blocks:
                logging.info(f"finetune block {index}")
                for param in layer.parameters():
                    param.requires_grad = True

    if local_rank == 0:
        check_preprocess_result(args)
        logging.info('finish the preprocess of docfeatures')
    dist.barrier()

    news_features, category_dict, subcategory_dict = read_news(args)
    logging.info('news_num:{}'.format(len(news_features)))

    #init the news_idx_incache and data_paths
    assert args.cache_num >= len(news_features)
    if local_rank == 0:
        idx = 0
        for news in news_features.keys():
            news_idx_incache[news] = [idx, -args.max_step_in_cache]
            idx += 1
        data_paths = get_files(dirname=os.path.join(args.root_data_dir,
                                                    'traindata'),
                               filename_pat=args.filename_pat)
        data_paths.sort()
        dump_args(args)
    dist.barrier()

    model = SpeedyFeed(args, bert_model, len(category_dict),
                       len(subcategory_dict))
    model = model.to(device)
    if args.world_size > 1:
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
    for ep in range(args.epochs):
        if local_rank == 0:
            # data_files.clear()
            while len(data_files) > 0:
                data_files.pop()
            data_files.extend(data_paths)
            random.seed(ep)
            random.shuffle(data_files)
        dist.barrier()

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

            bz_loss, encode_vecs = ddp_model(segments, token_masks, seg_masks,
                                             elements, cache_vec, batch_hist,
                                             batch_mask, batch_negs,
                                             key_position, fre_cnt)
            loss += bz_loss.data.float()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            #update the cache
            if args.drop_encoder_ratio > 0:
                encode_vecs = encode_vecs.detach().cpu().numpy()
                cache[update_cache] = encode_vecs

            optimizer.param_groups[0]['lr'] = args.pretrain_lr * warmup_linear(
                args, cnt + 1)  #* lr_scaler
            optimizer.param_groups[1]['lr'] = args.lr * warmup_linear(
                args, cnt + 1)  #* lr_scaler

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}] cost_time:{} step:{},  usernum: {}, train_loss: {:.5f}, lr:{}, pretrain_lr:{}'
                    .format(local_rank,
                            time.time() - start_time, cnt, usernum,
                            loss.data / (cnt + 1),
                            args.pretrain_lr * warmup_linear(args, cnt + 1),
                            args.lr * warmup_linear(args, cnt + 1)))

            # save model minibatch
            if local_rank == 0 and (cnt + 1) % args.save_steps == 0:
                ckpt_path = os.path.join(
                    args.model_dir, f'{args.savename}-epoch-{ep + 1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'category_dict': category_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

            dist.barrier()

        loss /= (cnt + 1)
        logging.info('epoch:{}, loss:{}, usernum:{}, time:{}'.format(
            ep + 1, loss, usernum,
            time.time() - start_time))

        # save model last of epoch
        if local_rank == 0:
            ckpt_path = os.path.join(
                args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
        logging.info("time:{}".format(time.time() - start_time))

    cleanup_process()
