import numpy as np
import pandas as pd
import glob
import torch
import h5py
import subprocess
from tqdm import tqdm
import os
import json
import sys

assert sys.platform in ['win32', 'linux'], 'platform not recognized'
if sys.platform == 'win32':
    NBC_ROOT = 'D:/nbc/'
else:
    NBC_ROOT = '/users/debert/data/datasets/nbc/'

def make_features(class_labels, n_classes, shift_constant=1.0):
    batch_size_, N_ = class_labels.size()
    f = torch.randn((batch_size_, N_, n_classes))
    shift = torch.zeros_like(f)
    shift.scatter_(2, class_labels.unsqueeze(2), shift_constant)
    return shift + f

def get_meta(sess):
    metapath = NBC_ROOT + 'meta/{}_start_end.json'.format(sess)
    if os.path.exists(metapath):
        with open(metapath) as f:
            d = json.load(f)
        return d['start'], d['end']
    else:
        spatial_ = pd.read_json(NBC_ROOT + 'raw/{}/spatial.json'.format(sess), orient='index')
        start, end = spatial_.iloc[0]['step'], spatial_.iloc[-1]['step']
        with open(metapath, 'w') as f:
            json.dump({'start': int(start), 'end': int(end)}, f)
        return start, end

def annotations_dataset(mode='train', subsample=45):
    paths = glob.glob(NBC_ROOT + 'tuning/*.txt')
    assert len(paths) > 0
    annotations = []
    for path in paths:
        sess = path.split('\\')[-1].replace('_annotations.txt', '')
        if mode == 'train':
            if sess[:4] in ['3_1b', '4_2b']:
                continue
        else:
            if sess[:4] not in ['3_1b', '4_2b']:
                continue
        annotations_ = pd.read_csv(path, header=None, names=['start_step', 'end_step', 'action', 'object'])
        start, end = get_meta(sess)
        annotations.append((annotations_, start, end))

    mapping = {'idle': 0, 'reach': 1, 'pick': 2, 'put': 3}
    N = 0
    for a, start, end in annotations:
        if end - start > N:
            N = end - start
        for action in a['action'].values:
            assert action in mapping, action

    labels = []
    lengths = []
    valid_classes = []
    for a, start, end in annotations:
        lengths.append(end - start)
        seq = np.zeros((N,))
        valid_classes_ = [0]
        for _, row in a.iterrows():
            label = mapping[row['action']]
            seq[row['start_step']-start:row['end_step']-start] = label
            if label not in valid_classes_:
                valid_classes_.append(label)
        labels.append(seq)
        valid_classes.append(valid_classes_)

    indices = list(range(0, N, subsample))
    labels = np.array(labels)[:, indices]
    lengths = np.array(lengths) // subsample

    labels = torch.LongTensor(labels)
    features = make_features(labels, len(mapping))
    lengths = torch.LongTensor(lengths)
    valid_classes = [torch.LongTensor(c) for c in valid_classes]
    return labels, features, lengths, valid_classes

def compute_max_k(labels):
    labels = labels.detach().cpu().numpy()
    prev = 0
    i = 0
    k = 0
    for seq in labels:
        for label in seq:
            if label == prev:
                i += 1
                if i > k:
                    k = i
            else:
                i = 0
    return k

def compute_baseline(labels):
    labels = labels.detach().cpu().numpy().flatten()
    return np.sum(labels == 0) / float(len(labels))

if __name__ == '__main__':
    labels, features, lengths, valid_classes = annotations_dataset('train')
    max_k = compute_max_k(labels)
    baseline = compute_baseline(labels)
    print(baseline, max_k)

    labels, features, lengths, valid_classes = annotations_dataset('test')
    max_k = compute_max_k(labels)
    baseline = compute_baseline(labels)
    print(baseline, max_k)
