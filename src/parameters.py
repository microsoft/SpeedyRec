# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from src import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train_test")

    #speedy
    parser.add_argument("--bucket_num", help="The number of buffer buckets", type=int, default=1)
    parser.add_argument("--cache_num", help="How many news representations can be stored in the cache", type=int, default=1250000)
    parser.add_argument("--drop_encoder_ratio", help="max-cache-rate", type=float, default=1)
    parser.add_argument("--max_step_in_cache", type=int, help='\gamma', default=20)
    parser.add_argument("--bus_connection", help="Whether to use the BusLM", type=utils.str2bool, default=True)
    parser.add_argument("--content_refinement", help="Whether to use the content refinement", type=utils.str2bool, default=True)
    parser.add_argument("--k2_for_BM25", help="the hyper-parameter of BM25", type=int, default=2)
    parser.add_argument("--max_keyword_freq", help="the max frequence of keywords, only required when use the content refinement", type=int, default=100)

    #data
    parser.add_argument("--root_data_dir", type=str, default="example_data/")
    parser.add_argument("--filename_pat", type=str, default="ProtoBuf_*.tsv")
    parser.add_argument("--model_dir", type=str, default='./model')
    parser.add_argument("--savename", type=str, default='speedy')
    parser.add_argument("--enable_prefetch", type=utils.str2bool, help='whether to prefetch data' ,default=True)
    parser.add_argument("--enable_prefetch_stream", type=utils.str2bool, default=False)
    parser.add_argument("--num_worker_preprocess", type=int, help='defautl to use half of cores in preprocess', default=-1)

    #model
    parser.add_argument("--user_log_length", type=int, default=100)
    parser.add_argument("--seg_length", type=int, default=32)
    parser.add_argument( "--news_query_vector_dim", type=int, default=200)
    parser.add_argument("--news_dim", type=int, default=64,)
    parser.add_argument("--word_embedding_dim", type=int, default=768,)
    parser.add_argument("--bert_layer_hidden", type=int, default=12,)
    parser.add_argument("--user_query_vector_dim", type=int, default=32,)
    parser.add_argument("--num_attention_heads", type=int, default=15,)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--add_pad", type=utils.str2bool, default=True)
    parser.add_argument("--use_pad", type=utils.str2bool, default=False)
    parser.add_argument("--body_seg_num", type=int, default=1)


    #negative sample
    parser.add_argument("--npratio", type=int, default=1)

    # model training
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pretrain_lr", type=float, default=0.00001)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--freeze_bert", type=utils.str2bool, default=False)
    parser.add_argument("--world_size", type=int, help='default to use all GPUs', default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--news_attributes",
        type=str,
        nargs='+',
        default=['title','abstract','body'],
        choices=['title', 'abstract', 'body', 'category', 'subcategory'])
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--warmup_step", type=int, default=5000)
    parser.add_argument("--schedule_step", type=int, default=25000)
    parser.add_argument("--save_steps", type=int, default=5000)

    # bert_model
    parser.add_argument("--bert_model",type=str,default='speedymodel',)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    parser.add_argument("--model_name_or_path", default="./example_data/pretrainedModel/base-uncased.bin", type=str,
                        help="Path to pre-trained model or shortcut name. ")
    parser.add_argument("--config_name", default="./example_data/pretrainedModel/uncased-config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="./example_data/pretrainedModel/uncased-vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--finetune_blocks",type=int,nargs='+',default=[],choices=list(range(12)))

    #test
    parser.add_argument("--load_ckpt_name",type=str,default = 'speedy-epoch-1.pt',help="choose which ckpt to load and test")

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
