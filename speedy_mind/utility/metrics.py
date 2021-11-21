from sklearn.metrics import roc_auc_score
import numpy as np
import logging
import torch

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

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


class MetricsDict():
    def __init__(self, metrics_name):
        self.metrics_dict = {}
        self.metrics_name = metrics_name

    def add_metric_dict(self, name):
        self.metrics_dict[name] = {}
        for metric_name in self.metrics_name:
            self.metrics_dict[name][metric_name] = []

    def cal_metrics(self, score, label):
        metric_rslt = {}
        if 'AUC' in self.metrics_name:
            metric_rslt["AUC"] = roc_auc_score(label, score)
        if 'MRR' in self.metrics_name:
            metric_rslt["MRR"] = mrr_score(label, score)
        if 'nDCG5' in self.metrics_name:
            metric_rslt["nDCG5"] = ndcg_score(label, score, k=5)
        if 'nDCG10' in self.metrics_name:
            metric_rslt["nDCG10"] = ndcg_score(label, score, k=10)
        return metric_rslt

    def update_metric_dict(self, name, metric_rslt):
        for metric_name in metric_rslt.keys():
            self.metrics_dict[name][metric_name].append(metric_rslt[metric_name])

    def print_metrics(self, local_rank, cnt, name):
        def __get_mean(arr):
            return [np.array(i).mean() for i in arr]

        arr = __get_mean([self.metrics_dict[name][metric_name] for metric_name in self.metrics_name])
        logging.info("[{}] {} Ed: {}: {}".format(
            local_rank, name, cnt,
            '\t'.join(["{:0.2f}".format(i * 100) for i in arr])))
