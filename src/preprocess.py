# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import os
import logging
from multiprocessing import Process,Manager,cpu_count
from nltk.corpus import stopwords
from transformers import BertTokenizer
from .refinement import content_refinement
from .utils import setuplogging
setuplogging()

def read_news(args,root_data_dir):
    '''
    load processed news
    Returns:
        news_features(dict):{news_id:(segment_ids,segment_mask,position_ids,frequence,elements)}
        category(dict):{category:id}
        subcategory(dict):{subcategory:id}
    '''
    if args.content_refinement:
        data_path = os.path.join(root_data_dir, 'processed_news_l{}_refine.pkl'.format(args.seg_length))
    else:
        data_path = os.path.join(root_data_dir, 'processed_news_l{}.pkl'.format(args.seg_length))

    assert os.path.exists(data_path)

    with open(data_path, 'rb') as f:
        process_news = pickle.load(f)
    return process_news['news_features'], process_news['category'], process_news['subcategory']

def check_preprocess_result(args,root_data_dir,mode='train',category=None,subcategory=None):
    if args.num_worker_preprocess == -1:
        preprocess_world_size = cpu_count()//2 - 1
    else:
        preprocess_world_size = args.num_worker_preprocess
    assert preprocess_world_size>0

    if args.content_refinement:
        data_path = os.path.join(root_data_dir, 'processed_news_l{}_refine.pkl'.format(args.seg_length))
    else:
        data_path = os.path.join(root_data_dir, 'processed_news_l{}.pkl'.format(args.seg_length))
    if not os.path.exists(data_path):
        if args.content_refinement and not os.path.exists(os.path.join(root_data_dir, 'refinement_k2={}.pkl'.format(args.k2_for_BM25))):
            logging.info('start content refinement')
            content_refinement(args,preprocess_world_size,root_data_dir)
        logging.info('start preprocess doc features')
        mul_prepocess(args,preprocess_world_size,root_data_dir,mode,category,subcategory)


def mul_prepocess(args,world_size,root_data_dir,mode='train',category=None,subcategory=None):
    '''
    process DocFeatures with multi-process
    '''
    # world_size = 5
    setuplogging()
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)

    manager = Manager()
    news_feature = manager.dict()
    categories = manager.dict()
    subcategories = manager.dict()

    if mode == 'test':
        for k,v in category.items():
            categories[k] = v
        for k,v in subcategory.items():
            subcategories[k] = v

    process_list = []
    for rank in range(world_size):
        p = Process(target=process_news, args=(rank,
                                               world_size,
                                               news_feature,
                                               categories,
                                               subcategories,
                                               root_data_dir,
                                               args,
                                               tokenizer,
                                               mode))
        p.start()
        process_list.append(p)
    logging.info('Waiting for all subprocesses done...')
    print('-----------------',world_size)

    for res in process_list:
        res.join()
    logging.info('All subprocesses done.')
    logging.info(f'news num:{len(news_feature)}')

    processed_news = {}
    processed_news['news_features'] = {}
    processed_news['category'] = {}
    processed_news['subcategory'] = {}
    for k, v in news_feature.items():
        processed_news['news_features'][k] = v
    for k, v in categories.items():
        processed_news['category'][k] = v
    for k, v in subcategories.items():
        processed_news['subcategory'][k] = v

    if args.content_refinement:
        save_path = os.path.join(root_data_dir,
                                 'processed_news_l{}_refine.pkl'.format(args.seg_length))
    else:
        save_path = os.path.join(root_data_dir, 'processed_news_l{}.pkl'.format(args.seg_length))

    print(len(processed_news['news_features']))
    with open(save_path, 'wb') as f:
        pickle.dump(processed_news, f)



def process_news(local_rank,
                 world_size,
                 news_feature,
                 categories,
                 subcategories,
                 data_path,
                 args,
                 tokenizer,
                 mode='train'):

    if args.content_refinement:
        refine_path =  os.path.join(data_path, 'refinement_k2={}.pkl'.format(args.k2_for_BM25))
        logging.info(f'loading {refine_path}')
        with open(refine_path, 'rb') as f:
            news_keywords = pickle.load(f)

        stpwords = stopwords.words('english')
        punctuation = ['!', ',', '.', '?','\\','-','|']
        stpwords.extend(punctuation)

    news_path = os.path.join(data_path,"DocFeatures.tsv")
    with open(news_path, "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % world_size != local_rank: continue

            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, body = splited

            tokens = []
            seg_mask = []
            key_position = []
            key_freq = []
            element = []

            if 'title' in args.news_attributes:
                if args.content_refinement:
                    cur_keys = news_keywords[doc_id][0]
                    cur_keys = [x[:2] for x in cur_keys if x[0] not in stpwords and len(x[0]) > 1][
                                  :args.seg_length]

                    input_ids, posi_id, freq, mask = refined_content_tokenizer(title,cur_keys,tokenizer,args.seg_length)

                else:
                    mask, input_ids = split_token(title, tokenizer, args.seg_length)
                    posi_id, freq = None, None

                tokens.append(input_ids)
                seg_mask.append(mask)
                key_position.append(posi_id)
                key_freq.append(freq)

            if 'abstract' in args.news_attributes:
                if args.content_refinement:
                    cur_keys = news_keywords[doc_id][1]
                    cur_keys = [x[:2] for x in cur_keys if x[0] not in stpwords and len(x[0]) > 1][
                               :args.seg_length]

                    input_ids, posi_id, freq, mask = refined_content_tokenizer(abstract, cur_keys, tokenizer, args.seg_length)

                else:
                    mask, input_ids = split_token(abstract, tokenizer, args.seg_length)
                    posi_id, freq = None, None

                tokens.append(input_ids)
                seg_mask.append(mask)
                key_position.append(posi_id)
                key_freq.append(freq)

            if 'body' in args.news_attributes:
                if args.content_refinement:
                    cur_keys = news_keywords[doc_id][2]
                    cur_keys = [x[:2] for x in cur_keys if x[0] not in stpwords and len(x[0]) > 1][
                               :args.seg_length]

                    input_ids, posi_id, freq, mask = refined_content_tokenizer(body, cur_keys, tokenizer,
                                                                            args.seg_length)

                else:
                    mask, input_ids = split_token(body, tokenizer, args.seg_length)
                    posi_id, freq = None, None

                tokens.append(input_ids)
                seg_mask.append(mask)
                key_position.append(posi_id)
                key_freq.append(freq)

            if 'category' in args.news_attributes:
                if mode == 'train':
                    if category not in categories:
                        categories[category] = len(categories)
                elif mode == 'test':
                    assert category in categories
                element.append(categories[category])

            if 'subcategory' in args.news_attributes:
                if mode == 'train':
                    if subcategory not in subcategories:
                        subcategories[subcategory] = len(subcategories)
                elif mode == 'test':
                    assert subcategory in subcategories
                element.append(subcategories[subcategory])

            news_feature[doc_id] = (tokens,seg_mask,key_position,key_freq,element)
    logging.info(f'Process: {local_rank} have finished')


def refined_content_tokenizer(text,
                              refined,
                              tokenizer,
                              seg_length):
    '''
    Process the refined content. If the token length of raw text is smaller than seg_length, we don't use the refined content.
    Args:
        text(str):raw text
        refined(list):the topk keywords generated from BM25
        tokenizer(BertTokenizer)
        seg_length(int)
    Returns:
        key_seg(list): the ids of tokens
        posi_id(list): position id
        fre(list): frequence of keyword
        m(int): 1 if len(text)>0 else 0
    '''
    all_ids = tokenizer(text)['input_ids']
    if len(all_ids) <= seg_length:
        key_seg = all_ids
        posi_id = list(range(len(all_ids)))
        freq = [0] + [1] * (len(all_ids) - 2) + [0]
        m = 1 if len(text) > 0 else 0
    else:
        key_seg, posi_id, fre, m = find_position(all_ids, refined, tokenizer, topkey=seg_length)
        key_seg = all_ids[:1] + key_seg[:seg_length - 2] + all_ids[-1:]
        posi_id = [0] + posi_id[:seg_length - 2] + [len(all_ids)]
        freq = [0] + fre[:seg_length - 2] + [0]
    return key_seg,posi_id,freq,m


def find_position(all_ids,entity,tokenizer,topkey=32):
    m = 0
    ent_seg = []
    posi_id = []
    fre = []

    if len(entity) > 0:
        m = 1
        ids_ent = {}
        for ent,qf in entity:
            ids = tokenizer(ent)['input_ids'][1:-1]
            if ids[0] not in ids_ent:
                ids_ent[ids[0]] = [(ids,qf)]
            else:
                ids_ent[ids[0]].append((ids,qf))

        i = 0
        while i < len(all_ids):
            id = all_ids[i]
            if id in ids_ent:
                for id_qf in ids_ent[id]:
                    ent_ids,qf = id_qf
                    if all_ids[i:i + len(ent_ids)] == ent_ids:
                        ids_ent[id].remove(id_qf)
                        ent_seg.extend(ent_ids)
                        # posi_id.extend([i] * len(ent_ids))
                        posi_id.extend(list(range(i,i+len(ent_ids))))
                        fre.extend([qf]*len(ent_ids))
                        i += (len(ent_ids) - 1)
                        break
            i += 1
            if len(posi_id)>=topkey:
                break
    return ent_seg,posi_id,fre,m


def split_token(text,tokenizer,max_l=32,reture_token_mask=False):
    if text == '':
        smask = 0
    else:
        smask = 1
    text = text.lower()
    if reture_token_mask:
        text = tokenizer(text, max_length=max_l,pad_to_max_length=True, truncation=True)
        return smask,text['input_ids'],text['attention_mask']
    else:
        text = tokenizer(text, max_length=max_l,pad_to_max_length=False, truncation=True)
        return smask,text['input_ids']

