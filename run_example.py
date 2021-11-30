# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
sys.path.append('./')
from src.parameters import parse_args
from src.train import *
from src.utils import *
from src.infer_news_vecs import mul_infer
from src.test_auc import ddp_test_auc
from src.test_recall import test_recall

if __name__ == "__main__":
    metrics = ['ranking','recall']
    args = parse_args()

    args.world_size = init_world_size(args.world_size) #get the number of GPU

    if 'train' in args.mode:
        ddp_train(args)

    if 'test' in args.mode:
        news_index, news_vecs = mul_infer(args)

        if 'ranking' in metrics:
            assert os.path.exists(os.path.join(args.root_data_dir,'testdata/impressions'))
            ddp_test_auc(args, news_index, news_vecs)
        if 'recall' in metrics:
            assert os.path.exists(os.path.join(args.root_data_dir,'testdata/daily_recall'))
            test_recall(args, news_index, news_vecs)

