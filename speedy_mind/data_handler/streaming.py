import os
import logging
import fnmatch
import random
import numpy as np
import tensorflow as tf
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import sys
import traceback

def get_files(dirname, filename_pat="*", recursive=False):
    if not tf.io.gfile.exists(dirname):
        logging.warning(f"no file in {dirname} !")
        return None
    files = []
    print(dirname)
    for x in tf.io.gfile.listdir(dirname):
        path = os.path.join(dirname, x)
        if tf.io.gfile.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    print()
    return files


def get_worker_files(dirnames,
                     worker_rank,
                     world_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    """Get file paths belong to one worker."""
    all_files = []

    for dirname in dirnames:
        all_files.extend(get_files(dirname, filename_pat))

    files = []
    for i in range(worker_rank, len(all_files), world_size):
        files.append(all_files[i])

    if shuffle:
        random.shuffle(files)

    logging.info(
        f"worker_rank:{worker_rank}, world_size:{world_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        path_len = len(data_paths)

        dataset = tf.data.Dataset.list_files(data_paths, shuffle=False).interleave(
            lambda x: tf.data.TextLineDataset(x).map(lambda y: tf.strings.join([y, x], separator="\t")),
            cycle_length=path_len,
            block_length=batch_size,
            # num_parallel_calls=min(path_len, batch_size),
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(3)
        self.next_batch = dataset.make_one_shot_iterator().get_next()
        self.session = None


    def reset(self):
        # print(f"StreamReader reset(), {self.session}, pid:{threading.currentThread()}")
        if self.session:
            self.session.close()
        self.session = tf.Session()
        self.endofstream = False

    def get_next(self):
        try:
            ret = self.session.run(self.next_batch)
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        # print(f"StreamReader reach_end(), {self.endofstream}")
        return self.endofstream


class StreamSampler:
    def __init__(
        self,
        data_dirs,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dirs,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.data_paths = data_paths
        self.stream_reader = StreamReader(data_paths, batch_size, enable_shuffle, shuffle_buffer_size)

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple) and not isinstance(next_batch, bytes):
            raise StopIteration
        # print(next_batch.shape)
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamReaderForSpeedy:
    def __init__(self, file, batch_size):
        self.file = file
        self.stream_reader = StreamReader(file, batch_size)

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple) and not isinstance(next_batch, bytes):
            raise StopIteration
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamSamplerTrainForSpeedyRec:
    def __init__(
        self,
        data_files,
        local_rank
    ):
        '''
        Args:
            data_files(manager.list()): the files storage train data
        '''
        files = []
        for i in range(local_rank, len(data_files), 8):
            files.append(data_files[i])
        self.data_files = files
        self.end = False
        self.sampler = None
        self.local_rank = local_rank

    def start_async(self):
        self.aval_count = 0
        self.end = False
        self.outputs = Queue(1000)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        try:
            self.sampler = self._generate_batch()
            for batch in self.sampler:
                if self.end:
                    break
                self.outputs.put(batch)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _generate_batch(self):
        while True:
            if len(self.data_files)>0:
                path = self.data_files.pop(0)
                with tf.io.gfile.GFile(path, "r") as f:
                    market = path.split("/")[-2]
                    for line in f:
                        yield line.strip('\n'), market
            else:
                self.end = True
                break

    def __iter__(self):
        self.join()
        self.start_async()
        return self

    def __next__(self):
        if self.sampler and  self.aval_count == 0 and self.end == True:
            raise StopIteration
        next_batch = self.outputs.get()
        self.outputs.task_done()
        self.aval_count -= 1
        return next_batch

    def join(self):
        self.end = True
        if self.sampler:
            while self.outputs.qsize() > 0:
                self.outputs.get()
                self.outputs.task_done()
            self.outputs.join()
            self.pool.shutdown(wait=True)
            logging.info("shut down pool.")
        self.sampler = None




class StreamReaderTest(StreamReader):
    def __init__(self, data_paths, batch_size, shuffle, shuffle_buffer_size=1000):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        # logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        path_len = len(data_paths)
        # logging.info(f"[StreamReader] path_len:{path_len}, paths: {data_paths}")
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x).map(lambda y: tf.strings.join([y, x], separator="\t")),
            cycle_length=path_len,
            block_length=batch_size,
            num_parallel_calls=min(path_len, batch_size),
        )

        # if shuffle:
        #     dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = dataset.make_one_shot_iterator().get_next()
        self.session = None


class StreamSamplerTest(StreamSampler):
    def __init__(
        self,
        data_dirs,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dirs,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.data_paths = data_paths
        self.stream_reader = StreamReaderTest(data_paths, batch_size, enable_shuffle, shuffle_buffer_size)

