# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import logging
import torch
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from LanguageModels.SpeedyModel import SpeedyModelForRec
from LanguageModels.configuration_tnlrv3 import TuringNLRv3Config

from .utils import (ndcg_score, mrr_score, ctr_score, setuplogging,
                    init_process, cleanup_process, init_config, dump_args)
from .dataloader import DataLoaderTest
from .speedyfeed import SpeedyFeed

from sklearn.metrics import roc_auc_score


def ddp_test_auc(args, news_index, news_vecs):
    logging.info('------start test auc------')
    mp.spawn(test_auc,
             args=(news_index, news_vecs, args),
             nprocs=args.world_size,
             join=True)


def load_model(args):
    checkpoint = torch.load(os.path.join(args.model_dir, args.load_ckpt_name))
    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']

    logging.info('loading model: {}'.format(args.bert_model))
    args, config = init_config(args, TuringNLRv3Config)
    bert_model = SpeedyModelForRec.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config)
    model = SpeedyFeed(args, bert_model, len(category_dict),
                       len(subcategory_dict))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def test_auc(local_rank, news_index, news_vecs, args):
    setuplogging()
    init_process(local_rank, args.world_size)

    if args.enable_gpu:
        os.environ["RANK"] = str(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    model = load_model(args)
    model = model.to(device)
    model.eval()

    if args.world_size > 1:
        ddp_model = DDP(model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True)
    else:
        ddp_model = model

    torch.set_grad_enabled(False)

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_vecs,
        news_bias_scoring=None,
        data_dir=os.path.join(args.root_data_dir, f'testdata/impressions'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=args.world_size,
        worker_rank=local_rank,
        cuda_device_idx=local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )

    AUC = [[], []]
    MRR = [[], []]
    nDCG5 = [[], []]
    nDCG10 = [[], []]
    CTR1 = [[], []]
    CTR3 = [[], []]
    CTR5 = [[], []]
    CTR10 = [[], []]

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
                                              '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    for cnt, (log_vecs, log_mask, news_vecs,
              labels) in tqdm(enumerate(dataloader)):
        his_lens = torch.sum(log_mask,
                             dim=-1).to(torch.device("cpu")).detach().numpy()

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)

        if args.world_size > 1:
            user_vecs = ddp_model.module.user_encoder.infer_user_vec(
                log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()
        else:
            user_vecs = ddp_model.user_encoder.infer_user_vec(
                log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for index, user_vec, news_vec, label, his_len in zip(
                range(len(labels)), user_vecs, news_vecs, labels, his_lens):

            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(news_vec, user_vec)

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)
            ctr1 = ctr_score(label, score, k=1)
            ctr3 = ctr_score(label, score, k=3)
            ctr5 = ctr_score(label, score, k=5)
            ctr10 = ctr_score(label, score, k=10)

            AUC[0].append(auc)
            MRR[0].append(mrr)
            nDCG5[0].append(ndcg5)
            nDCG10[0].append(ndcg10)
            CTR1[0].append(ctr1)
            CTR3[0].append(ctr3)
            CTR5[0].append(ctr5)
            CTR10[0].append(ctr10)

            if his_len <= 5:
                AUC[1].append(auc)
                MRR[1].append(mrr)
                nDCG5[1].append(ndcg5)
                nDCG10[1].append(ndcg10)
                CTR1[1].append(ctr1)
                CTR3[1].append(ctr3)
                CTR5[1].append(ctr5)
                CTR10[1].append(ctr10)

        if cnt == 0:
            for i in range(2):
                print_metrics(
                    local_rank, 0,
                    get_mean([
                        AUC[i], MRR[i], nDCG5[i], nDCG10[i], CTR1[i], CTR3[i],
                        CTR5[i], CTR10[i]
                    ]))
        if (cnt + 1) % args.log_steps == 0:
            for i in range(2):
                print_metrics(local_rank, (cnt + 1) * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                                                                               nDCG10[i], CTR1[i], CTR3[i], CTR5[i],
                                                                               CTR10[i]]))
    dataloader.join()

    for i in range(2):
        print_metrics(local_rank, (cnt + 1) * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                                                                       nDCG10[i], CTR1[i], CTR3[i], CTR5[i],
                                                                       CTR10[i]]))
    cleanup_process()
