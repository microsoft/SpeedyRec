# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import logging
import random
from pathlib import Path
import numpy as np
import sys
import traceback

import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from utility.utils import (setuplogger, init_process, cleanup_process, warmup_linear, get_device, lr_schedule,
                            get_barrier, only_on_main_process, check_args_environment, dump_args)
from utility.metrics import acc, MetricsDict
from parameters import parse_args

from data_handler.streaming import get_files
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TrainDataloader import DataLoaderTrainForSpeedyRec
from data_handler.TestDataloader import DataLoaderTest

from models.speedyrec import MLNR


def ddp_train_vd(args):
    '''
    Distributed training
    '''
    setuplogger()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args = check_args_environment(args)

    logging.info('-----------start train------------')

    cache_state = mp.Manager().dict()
    data_files = mp.Manager().list([])
    end_dataloder = mp.Manager().Value('b', False)
    end_train = mp.Manager().Value('b', False)
    mp.spawn(train,
             args=(
             args, cache_state, data_files, end_dataloder, end_train),
             nprocs=args.world_size,
             join=True)


def train(local_rank,
          args,
          cache_state,
          data_files,
          end_dataloder,
          end_train,
          dist_training=True):

    setuplogger()
    try:
        if dist_training:
            init_process(local_rank, args.world_size)
        device = get_device()
        barrier = get_barrier(dist_training)

        news_info, news_combined = get_news_feature(args, mode='train')
        with only_on_main_process(local_rank, barrier) as need:
            if need:
                data_paths = []
                data_dirs = os.path.join(args.root_data_dir, 'train/')
                data_paths.extend(get_files(data_dirs, args.filename_pat))
                data_paths.sort()

        model = MLNR(args)
        if 'speedymind_ckpts' in args.pretrained_model_path:
            ckpt = torch.load(os.path.join(args.pretrained_model_path, 'pytorch_model.bin'))
            model.load_state_dict(ckpt['model_state_dict'])

        model = model.to(device)
        rest_param = filter(
            lambda x: id(x) not in list(map(id, model.news_encoder.unicoder.parameters())),
            model.parameters())
        optimizer = optim.Adam([{
            'params': model.news_encoder.unicoder.parameters(),
            'lr': args.pretrain_lr  #lr_schedule(args.pretrain_lr, 1, args)
        }, {
            'params': rest_param,
            'lr': args.lr  #lr_schedule(args.lr, 1, args)
        }])
        #

        if dist_training:
            ddp_model = DDP(model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=True)
        else:
            ddp_model = model

        logging.info('Training...')
        start_time = time.time()
        test_time = 0.0
        global_step = 0
        best_count = 0
        optimizer.zero_grad()

        loss = 0.0
        best_auc = 0.0
        accuary = 0.0
        hit_num = 0
        all_num = 1
        encode_num = 0
        cache = np.zeros((len(news_combined),args.news_dim))
        for ep in range(args.epochs):
            with only_on_main_process(local_rank, barrier) as need:
                if need:
                    while len(data_files) > 0:
                        data_files.pop()
                    data_files.extend(data_paths)
                    random.shuffle(data_files)
            barrier()

            dataloader = DataLoaderTrainForSpeedyRec(
                args=args,
                data_files=data_files,
                cache_state=cache_state,
                end=end_dataloder,
                local_rank=local_rank,
                world_size=args.world_size,
                news_features=news_combined,
                news_index=news_info.news_index,
                enable_prefetch=args.enable_prefetch,
                enable_prefetch_stream=args.enable_prefetch_stream,
                global_step=global_step,
                add_pad_news=True)

            ddp_model.train()
            pad_doc = torch.zeros(1, args.news_dim, device=device)

            for cnt, batch in tqdm(enumerate(dataloader)):
                with torch.autograd.set_detect_anomaly(True):
                    address_cache, update_cache, satrt_inx, end_inx, batch = batch
                    global_step += 1

                    if args.enable_gpu:
                        input_ids, hist_sequence, hist_sequence_mask, candidate_inx, label_batch = (
                            x.cuda(device=device,non_blocking=True) if x is not None else x
                            for x in batch[:5])
                    else:
                        input_ids, hist_sequence, hist_sequence_mask, candidate_inx, label_batch = batch[:5]

                    encode_num += input_ids.size(0)

                    # Get news vecs from cache.
                    if address_cache is not None:
                        # cache_vec = [cache[inx] for inx in address_cache]
                        cache_vec = cache[address_cache]
                        cache_vec = torch.FloatTensor(
                            cache_vec).cuda(device=device, non_blocking=True)

                        # atime += time.time() - temp_stime
                        hit_num += cache_vec.size(0)
                        all_num += cache_vec.size(0)

                    else:
                        cache_vec = None
                        hit_num += 0

                    if cache_vec is not None:
                        cache_vec = torch.cat([pad_doc, cache_vec], 0)
                    else:
                        cache_vec = pad_doc

                    if input_ids.size(0) > 0:
                        if dist_training:
                            encode_vecs = ddp_model.module.news_encoder(input_ids)
                        else:
                            encode_vecs = ddp_model.news_encoder(input_ids)
                    else:
                        encode_vecs = None

                    all_tensors = [torch.empty_like(encode_vecs) for _ in range(args.world_size)]
                    dist.all_gather(all_tensors, encode_vecs)
                    all_tensors[local_rank] = encode_vecs
                    all_encode_vecs = torch.cat(all_tensors, dim=0)
                    news_vecs = torch.cat([cache_vec, all_encode_vecs], 0)

                    all_num += all_encode_vecs.size(0)
                    bz_loss, y_hat = ddp_model(news_vecs,
                                             hist_sequence, hist_sequence_mask,
                                             candidate_inx,
                                             label_batch)

                    loss += bz_loss.item()
                    bz_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    accuary += acc(label_batch, y_hat)

                    # update the cache
                    if args.max_step_in_cache > 0 and encode_vecs is not None:
                        update_vecs = all_encode_vecs.detach().cpu().numpy()[:len(update_cache)]
                        cache[update_cache] = update_vecs

                    optimizer.param_groups[0]['lr'] = lr_schedule(args.pretrain_lr, global_step, args)
                    optimizer.param_groups[1]['lr'] = lr_schedule(args.lr, global_step, args)

                    barrier()

                if global_step % args.log_steps == 0:
                    logging.info(
                        '[{}] cost_time:{} step:{}, train_loss: {:.5f}, acc:{:.5f}, hit:{}, encode_num:{}, lr:{:.8f}, pretrain_lr:{:.8f}'.format(
                            local_rank, time.time() - start_time-test_time, global_step, loss / args.log_steps, accuary / args.log_steps, hit_num/all_num, encode_num,
                            optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))
                    loss = 0.0
                    accuary = 0.0

                if global_step%args.test_steps == 0 and local_rank == 0:
                    stest_time = time.time()
                    auc = test(model, args, device, news_info.category_dict, news_info.subcategory_dict)
                    ddp_model.train()
                    logging.info('step:{}, auc:{}'.format(global_step, auc))
                    test_time = test_time + time.time()-stest_time

                # save model minibatch
                if local_rank == 0 and global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep + 1}-{global_step}.pt')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'category_dict': news_info.category_dict,
                            'subcategory_dict': news_info.subcategory_dict,
                        }, ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")

            logging.info('epoch:{}, time:{}, encode_num:{}'.format(ep + 1, time.time() - start_time-test_time, encode_num))
            # save model after an epoch
            if local_rank == 0:
                ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'category_dict': news_info.category_dict,
                        'subcategory_dict': news_info.subcategory_dict,
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                auc = test(model, args, device, news_info.category_dict, news_info.subcategory_dict)
                ddp_model.train()

                if auc>best_auc:
                    best_auc = auc
                else:
                    best_count += 1
                    if best_auc >= 3:
                        logging.info("best_auc:{}, best_ep:{}".format(best_auc, ep-3))
                        end_train.value = True
            barrier()
            if end_train.value:
                break

        if dist_training:
            cleanup_process()

    except:
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)



def test(model, args, device, category_dict, subcategory_dict):
    model.eval()

    with torch.no_grad():
        news_info, news_combined = get_news_feature(args, mode='dev', category_dict=category_dict, subcategory_dict=subcategory_dict)
        news_vecs = infer_news(model, device, news_combined)

        dataloader = DataLoaderTest(
            news_index=news_info.news_index,
            news_scoring=news_vecs,
            data_dirs=[os.path.join(args.root_data_dir,
                                    f'dev/')],
            filename_pat=args.filename_pat,
            args=args,
            world_size=1,
            worker_rank=0,
            cuda_device_idx=0,
            enable_prefetch=args.enable_prefetch,
            enable_shuffle=args.enable_shuffle,
            enable_gpu=args.enable_gpu,
        )

        results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
        results.add_metric_dict('all users')
        results.add_metric_dict('cold users')

        for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
            his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True).to(torch.device("cpu")).detach().numpy()

            for index, user_vec, news_vec, label, his_len in zip(
                    range(len(labels)), user_vecs, news_vecs, labels, his_lens):

                if label.mean() == 0 or label.mean() == 1:
                    continue
                score = np.dot(
                    news_vec, user_vec
                )

                metric_rslt = results.cal_metrics(score, label)
                results.update_metric_dict('all users', metric_rslt)

                if his_len <= 5:
                    results.update_metric_dict('cold users', metric_rslt)

            # if cnt % args.log_steps == 0:
            #     results.print_metrics(0, cnt * args.batch_size, 'all users')
            #     results.print_metrics(0, cnt * args.batch_size, 'cold users')

        dataloader.join()
        for i in range(2):
            results.print_metrics(0, cnt * args.batch_size, 'all users')
            results.print_metrics(0, cnt * args.batch_size, 'cold users')

    return np.mean(results.metrics_dict["all users"]['AUC'])


if __name__ == '__main__':
    setuplogger()
    args = parse_args()
    ddp_train_vd(args)



