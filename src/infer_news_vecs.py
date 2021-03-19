# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.multiprocessing import Pool, set_start_method

from LanguageModels.SpeedyModel import SpeedyModelForRec
from LanguageModels.configuration_tnlrv3 import TuringNLRv3Config

from .utils import init_config, setuplogging
from .preprocess import read_news, check_preprocess_result
from .speedyfeed import SpeedyFeed


def pad_to_fix_len(x, fix_length, padding_value=0, padding_front=True):
    if padding_front:
        pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
    else:
        pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length-len(x))
    return pad_x,mask


def news_feature_batch(news_feature,
                       batch_size,
                       seg_num,
                       seq_l,
                       content_refinement=True,
                       max_keyword_freq=100):
    segments, token_masks, seg_masks,key_position,fre_cnt, elements = [],[],[],[],[],[]
    for i in range(len(news_feature)):
        seg, s_mask, position, fre, ele = news_feature[i]
        for j in range(seg_num):
            s,m = pad_to_fix_len(seg[j],fix_length=seq_l,padding_front=False)
            segments.append(s)
            token_masks.append(m)
            if content_refinement:
                p,_ = pad_to_fix_len(position[j],fix_length=seq_l,padding_front=False)
                key_position.append(p)
                f,_ = pad_to_fix_len(fre[j],fix_length=seq_l,padding_front=False)
                f = [min(x, max_keyword_freq-1) for x in f]
                fre_cnt.append(f)

        seg_masks.append(s_mask[:seg_num])
        elements.append(ele)
        if (i+1)%batch_size == 0:
            yield segments, token_masks, seg_masks, key_position,fre_cnt, elements
            segments, token_masks, seg_masks, key_position, fre_cnt, elements = [], [], [], [], [], []

    if len(seg_masks)>0:
        yield segments, token_masks, seg_masks, key_position, fre_cnt, elements


def sigle_process_infer(local_rank, local_features, checkpoint,
          args):
    '''
    Args:
        local_rank(int): the rank of current process
        args: parameters
    '''
    if args.enable_gpu:
        os.environ["RANK"] = str(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    args, config = init_config(args,TuringNLRv3Config)
    if args.pretrained_model_path != 'None':
        bert_model = SpeedyModelForRec.from_pretrained(
            args.pretrained_model_path,
            from_tf=bool('.ckpt' in args.pretrained_model_path),
            config=config)
    else:
        bert_model = SpeedyModelForRec(config)

    model = SpeedyFeed(args, bert_model, len(checkpoint["category_dict"]), len(checkpoint["subcategory_dict"]))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    news_vecs = []
    batch_size = args.batch_size//(args.seg_length)
    with torch.no_grad():
        d = tqdm(news_feature_batch(local_features, batch_size, args.seg_num, args.seg_length),
                 total=(int(len(local_features) / batch_size)))
        for input_ids in d:
            segments, token_masks, seg_masks, key_position, fre_cnt, elements = input_ids
            segments = torch.LongTensor(segments).to(device)
            token_masks = torch.FloatTensor(token_masks).to(device)
            seg_masks = torch.FloatTensor(seg_masks).to(device)
            if args.content_refinement:
                key_position = torch.LongTensor(key_position).to(device)
                fre_cnt = torch.LongTensor(fre_cnt).to(device)
            else:
                key_position, fre_cnt = None, None
            elements = torch.LongTensor(elements).to(device)

            vec = model.news_encoder(segments, token_masks, seg_masks, key_position, fre_cnt, elements)
            vec = vec.to(torch.device("cpu")).detach().numpy()
            news_vecs.extend(vec)

    news_vecs = np.array(news_vecs)
    return news_vecs


def mul_infer(args):
    setuplogging()
    set_start_method('spawn',force=True)
    root_data_dir = os.path.join(args.root_data_dir,'testdata')

    checkpoint = torch.load(os.path.join(args.model_dir, args.load_ckpt_name),map_location=torch.device('cpu'))
    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    logging.info('load ckpt: {}'.format(args.load_ckpt_name))

    check_preprocess_result(args,
                            root_data_dir,
                            mode='test',
                            category=category_dict,
                            subcategory=subcategory_dict)
    logging.info('finish the preprocess of docfeatures')

    docid_features, category_dict, subcategory_dict = read_news(args,root_data_dir)

    news_index = {}
    news_feature = []
    cnt = 0
    for k, v in docid_features.items():
        news_index[k] = cnt
        news_feature.append(v)
        cnt += 1

    news_num = len(news_feature)
    logging.info('news_num:{}'.format(news_num))

    pool = Pool(processes=args.world_size)
    results = []
    sigle_size = news_num//args.world_size
    for rank in range(args.world_size):
        start = sigle_size*rank
        end = min(sigle_size*(rank+1),news_num)
        local_features = news_feature[start:end]
        result = pool.apply_async(sigle_process_infer, args=(rank, local_features, checkpoint, args))
        results.append(result)
    pool.close()
    pool.join()

    results = [x.get() for x in results]
    news_vecs = np.concatenate(results,0)
    return news_index, news_vecs


