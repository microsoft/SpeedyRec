# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import torch
import numpy as np
import zipfile

from utility.utils import (setuplogger,  dump_args, check_args_environment)
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TestDataloader import DataLoaderLeader
from models.speedyrec import MLNR

def generate_submission(args):
    setuplogger()
    args = check_args_environment(args)
    logging.info('-----------start test------------')

    local_rank = 0
    device = torch.device('cuda', int(local_rank))

    model = MLNR(args)
    model = model.to(device)
    ckpt = torch.load(args.load_ckpt_name)
    model.load_state_dict(ckpt['model_state_dict'])

    prediction(model, args, device, ckpt['category_dict'], ckpt['subcategory_dict'])


def prediction(model, args, device, category_dict, subcategory_dict):
    model.eval()
    with torch.no_grad():
        news_info, news_combined = get_news_feature(args, mode='test', category_dict=category_dict,
                                                    subcategory_dict=subcategory_dict)
        news_vecs = infer_news(model, device, news_combined)

        dataloader = DataLoaderLeader(
            news_index=news_info.news_index,
            news_scoring=news_vecs,
            data_dirs=os.path.join(args.root_data_dir,
                                    f'test/'),
            filename_pat=args.filename_pat,
            args=args,
            world_size=1,
            worker_rank=0,
            cuda_device_idx=0,
            enable_prefetch=False,
            enable_shuffle=args.enable_shuffle,
            enable_gpu=args.enable_gpu,
        )

        f = open('prediction.txt', 'w', encoding='utf-8')
        for cnt, (impids, log_vecs, log_mask, candidate_vec) in enumerate(dataloader.generate_batch()):

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True).to(torch.device("cpu")).detach().numpy()

            for id, user_vec, news_vec in zip(
                    impids, user_vecs, candidate_vec):

                score = np.dot(
                    news_vec, user_vec
                )
                pred_rank = (np.argsort(np.argsort(score)[::-1]) + 1).tolist()
                f.write(str(id) + ' ' + '[' + ','.join([str(x) for x in pred_rank]) + ']' + '\n')

        f.close()

    zip_file = zipfile.ZipFile('prediction.zip', 'w')
    zip_file.write('prediction.txt')
    zip_file.close()
    os.remove('prediction.txt')

if __name__ == "__main__":
    from parameters import parse_args
    setuplogger()
    args = parse_args()
    generate_submission(args)
