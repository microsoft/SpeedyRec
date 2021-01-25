# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.multiprocessing as mp
from src.parameters import parse_args
from pathlib import Path
from src.train import *
from src.utils import *

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if 'train' in args.mode:
        print('-----------trian------------')
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
            train(0, args, cache, news_idx_incache, prefetch_step, end, data_files)
