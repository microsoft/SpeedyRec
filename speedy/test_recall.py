# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import numpy as np
import hnswlib
import random
import os
import torch
import logging
from tqdm import tqdm
from .test_auc import load_model
from .infer_news_vecs import pad_to_fix_len

def get_day_item(data_dir, date, news_index, pad_news_index):
    filename = os.path.join(data_dir, "daily_news_{}.tsv".format(date))
    with open(filename, 'r', encoding='utf-8') as f:
        day_item = [news_index[x] if x in news_index else pad_news_index for x in f.read().strip().split('\t')]
    return day_item


def CreatIndex(batchsize, all_item_vec, itemid, mode):
    item_num = np.shape(all_item_vec)[0]
    p = hnswlib.Index(space=mode, dim=np.shape(all_item_vec)[-1])
    p.init_index(max_elements=item_num, ef_construction=200, M=100)
    p.set_ef(1500)

    for step in range(math.ceil(item_num / batchsize)):
        start = step * batchsize
        end = min((step + 1) * batchsize, item_num)
        batch_array = all_item_vec[start:end]
        p.add_items(batch_array, itemid[start:end])
    return p


def hist_pos(data_dir, date, news_index, user_log_length):
    filename = os.path.join(data_dir, "history_positive_{}.tsv".format(date))
    history = []
    mask = []
    positems = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            linesplit = line.strip().split('\t')
            uid, hist, pos = linesplit
            temp_hist = [news_index[x] for x in hist.split(';')]
            temp_hist, temp_mask = pad_to_fix_len(temp_hist,user_log_length)
            history.append(temp_hist)
            mask.append(temp_mask)
            positems.append([news_index[x] for x in pos.split(';')])
    return history, mask, positems


def generate_user_data(history, mask, all_item_embedding, user_batch_size=512):
    # history = np.array(history)
    step = math.ceil(len(history) / user_batch_size)
    for i in range(step):
        start = user_batch_size * i
        end = min(user_batch_size * (i + 1), len(history))
        index = history[start:end]
        index = np.array(index)
        batch_mask = mask[start:end]
        yield all_item_embedding[index], batch_mask


def CreatUserEmbed(history, mask, all_item_embedding, user_batch_size, model, device):
    user_embedding = []
    with torch.no_grad():
        user_progress = tqdm(enumerate(generate_user_data(history, mask, all_item_embedding, user_batch_size)),
                             dynamic_ncols=True,
                             total=(math.ceil(len(history) / user_batch_size)))
        for step, batch in user_progress:
            log_vecs, log_mask = batch
            log_vecs = torch.from_numpy(log_vecs).cuda(non_blocking=True).float().to(device)
            log_mask = torch.Tensor(log_mask).cuda(non_blocking=True).float().to(device)

            user_vecs = model.user_encoder.infer_user_vec(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()
            user_embedding.extend(user_vecs)

    user_embedding = np.array(user_embedding)
    return user_embedding


def get_result(user_embedding, positems, p, hnswlib_batch_size=5000):
    recall20 = 0
    recall50 = 0
    recall100 = 0
    recall200 = 0
    recall500 = 0

    all_ans = []
    for step in range(math.ceil(len(user_embedding) / hnswlib_batch_size)):
        start = step * hnswlib_batch_size
        end = min((step + 1) * hnswlib_batch_size, len(user_embedding))
        batch_array = user_embedding[start:end]
        ans, dis = p.knn_query(batch_array, k=200)
        all_ans.extend(ans)

    user_num = len(user_embedding)
    for i in range(user_num):
        ans = all_ans[i]

        pos = set(positems[i])

        ans200 = set(ans[:200])
        ans100 = set(ans[:100])
        ans50 = set(ans[:50])
        ans20 = set(ans[:20])

        recall20 += len(set.intersection(pos, ans20)) / len(pos)
        recall50 += len(set.intersection(pos, ans50)) / len(pos)
        recall100 += len(set.intersection(pos, ans100)) / len(pos)
        recall200 += len(set.intersection(pos, ans200)) / len(pos)

    recall20 = recall20 / user_num
    recall50 = recall50 / user_num
    recall100 = recall100 / user_num
    recall200 = recall200 / user_num

    return np.array([recall20, recall50, recall100, recall200])


def consine_similarity(item_embedding):
    num = item_embedding.shape[0]
    norm = np.linalg.norm(item_embedding, axis=1)
    norm = np.dot(np.expand_dims(norm, 1), np.expand_dims(norm, 0))
    simi = np.dot(item_embedding, np.transpose(item_embedding))
    simi = simi / (norm + 1e-6)
    simi = simi * (1 - np.eye(num, num))
    return np.sum(simi) / (num ** 2 - num)


def get_similarity(item_embedding):
    pair_num = 10
    sample_num = 10000
    allitems = list(range(1, len(item_embedding) - 1))
    cos = 0
    for i in range(sample_num):
        items = random.sample(allitems, pair_num)
        cos += consine_similarity(item_embedding[items])
    cos = cos / sample_num
    return cos


def test_recall(args,news_index,news_embed):
    logging.info('------start test recll------')
    user_batch_size = args.test_batch_size
    hnswlib_batch_size = 5000
    mode = 'ip'  # 'cosine'

    device = torch.device("cuda") if args.enable_gpu else torch.device("cpu")
    model = load_model(args)
    model.to(device)
    res = np.array([0.0] * 4)
    simi = get_similarity(news_embed)

    pad_news_index = len(news_embed)
    pad_news = np.zeros((1,news_embed.shape[-1]))
    news_embed = np.concatenate([news_embed,pad_news],0)

    date_recall = list(range(1,32))
    data_dir = os.path.join(args.root_data_dir,'testdata/daily_recall')

    for date in date_recall:
        day_item = get_day_item(data_dir,date,news_index,pad_news_index)
        item_num = len(day_item)

        item_embedding = news_embed[day_item]
        p = CreatIndex(hnswlib_batch_size, item_embedding, day_item, mode)
        p.set_ef(1500)

        history, mask, positems = hist_pos(data_dir,date,news_index,args.user_log_length)

        user_embedding = CreatUserEmbed(history, mask, news_embed, user_batch_size, model, device)
        user_num = len(positems)
        logging.info('recall_data: {} user_num:{}, item_num:{}'.format(date,user_num,item_num))

        day_res = get_result(user_embedding, positems, p, hnswlib_batch_size)
        res += day_res

        info = '{}-: recall20:{},recall50:{},recall100:{},recall200:{}'.format(date,
                                                                              day_res[0], day_res[1],
                                                                              day_res[2], day_res[3])
        logging.info(info)

    res = res / len(date_recall)
    info = 'Avg. simi:{} recall20:{},recall50:{},recall100:{},recall200:{}'.format(simi,
                                                                                   res[0], res[1],
                                                                                   res[2], res[3])
    logging.info(info)