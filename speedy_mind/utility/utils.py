import logging
import os
import sys
import torch
import numpy as np
import argparse
import random
from contextlib import contextmanager
import torch.distributed as dist

from transformers import AutoTokenizer, AutoConfig, AutoModel
from models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

MODEL_CLASSES = {
    'unilm': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    'others': (AutoConfig, AutoModel, AutoTokenizer)
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def setuplogger():
    root = logging.getLogger()
    # logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO)
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'

    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )
    torch.cuda.set_device(rank)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def cleanup_process():
    dist.destroy_process_group()


def get_device():
    if torch.cuda.is_available():
        local_rank = os.environ.get("RANK", 0)
        return torch.device('cuda', int(local_rank))
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

def warmup_linear(args, step):
    if step <= args.warmup_step:
        return step/args.warmup_step
    return max(1e-4,(args.schedule_step-step)/(args.schedule_step-args.warmup_step))

def lr_schedule(init_lr, step, args):
    if args.warmup:
        return warmup_linear(args, step)*init_lr
    else:
        return init_lr

def init_world_size(world_size):
    assert world_size <= torch.cuda.device_count()
    return torch.cuda.device_count() if world_size == -1 else world_size

def check_args_environment(args):
    if not torch.cuda.is_available():
        logging.warning("Cuda is not available, " \
                        "related options will be disabled")
    args.enable_gpu = torch.cuda.is_available() & args.enable_gpu
    return args


class timer:
    """
    Time context manager for code block
    """
    from time import time
    NAMED_TAPE = {}

    def __init__(self, name, **kwargs):
        self.name = name
        if name not in timer.NAMED_TAPE:
            timer.NAMED_TAPE[name] = 0

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        timer.NAMED_TAPE[self.name] += timer.time() - self.start
        print(self.name, timer.NAMED_TAPE[self.name])


