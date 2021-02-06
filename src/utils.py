# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import sys
import torch
import numpy as np
import argparse
import re
import os
import torch.distributed as dist
from contextlib import contextmanager

def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setuplogging():
    from .world import LOG_LEVEL

    root = logging.getLogger()
    # logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO)
    root.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()
    root.addHandler(handler)


def init_process(rank, world_size):
    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,)
    torch.cuda.set_device(rank)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup_process():
    dist.destroy_process_group()

def get_device():
    if torch.cuda.is_available():
        local_rank = os.environ.get("RANK", 0)
        return torch.device('cuda', local_rank)
    return torch.device('cpu')

def get_barrier(dist_training):
    if dist_training:
        return dist.barrier
    def do_nothing():
        pass
    return do_nothing
 
@contextmanager
def only_on_main_process(local_rank, barrier):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    need = True
    if local_rank not in [-1, 0]:
        barrier()
        need = False
    yield need
    if local_rank == 0:
        barrier()

def init_config(args,Configclass):
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
    config = Configclass.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)

    seg_num = 0
    for name in args.news_attributes:
        if name == 'title':
            seg_num += 1
        elif name == 'abstract':
            seg_num += 1
        elif name == 'body':
            seg_num += args.body_seg_num
    args.seg_num = seg_num

    if seg_num>1 and args.bus_connection:
        args.bus_num = seg_num
    else:
        args.bus_num = 0

    config.bus_num = args.bus_num
    config.hidden_size = args.word_embedding_dim
    config.num_hidden_layers = args.bert_layer_hidden

    return args,config


def warmup_linear(args,step):
    if step <= args.warmup_step:
        return step/args.warmup_step
    return max(1e-4,(args.schedule_step-step)/(args.schedule_step-args.warmup_step))


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")

def check_args_environment(args):
    if not torch.cuda.is_available():
        logging.warning("Cuda is not available, " \
                        "related options will be disabled")
    args.enable_gpu = torch.cuda.is_available() & args.enable_gpu
    
    return args

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None


