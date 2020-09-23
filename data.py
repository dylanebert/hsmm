import numpy as np
import pandas as pd
import torch
import os
import sys
import random
import argparse
import glob
import json
import nbc

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT envvar'
NBC_ROOT = os.environ['NBC_ROOT']

def add_args(parser):
    nbc.add_args(parser)
    parser.add_argument('--dataset', choices=['toy', 'nbc', 'nbc_annotations', 'nbc_synthetic', 'unit_test'], default='toy')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--n_train', help='number of train instances (for generated datasets)', type=int, default=100)
    parser.add_argument('--n_test', help='number of test instances (for generated datasets)', type=int, default=10)
    parser.add_argument('--n_classes', help='number of classes (for fully unsupervised datasets)', type=int, default=5)
    parser.add_argument('--max_k', help='max state length', type=int, default=20)
    parser.add_argument('--unit_test_dim', help='features dimensionality (for unit_test dataset)', type=int, default=1)

def dataset_from_args(args):
    if args.dataset == 'toy':
        train_dset = Dataset(*synthetic_data(args.n_train), max_k=args.max_k, device=torch.device(args.device))
        test_dset = Dataset(*synthetic_data(args.n_test), max_k=args.max_k, device=torch.device(args.device))
    elif args.dataset == 'nbc':
        train_dset = UnsupervisedDataset(*nbc.NBCData(args, 'train').to_dataset(), n_classes=args.n_classes, device=torch.device(args.device))
        test_dset = UnsupervisedDataset(*nbc.NBCData(args, 'test').to_dataset(), n_classes=args.n_classes, device=torch.device(args.device))
    elif args.dataset == 'nbc_annotations':
        train_dset = Dataset(*nbc_annotations_dataset('train'), max_k=args.max_k, device=torch.device(args.device))
        test_dset = Dataset(*nbc_annotations_dataset('test'), max_k=args.max_k, device=torch.device(args.device))
    elif args.dataset == 'unit_test':
        train_dset = Dataset(*unit_test_data(args.n_train, n_dim=args.unit_test_dim), n_classes=2, max_k=args.max_k, device=torch.device(args.device))
        test_dset = Dataset(*unit_test_data(args.n_test, n_dim=args.unit_test_dim), n_classes=2, max_k=args.max_k, device=torch.device(args.device))
    else:
        assert args.dataset == 'nbc_synthetic'
        train_dset = Dataset(*nbc_synthetic_dataset(args.n_train), max_k=args.max_k, device=torch.device(args.device))
        test_dset = Dataset(*nbc_synthetic_dataset(args.n_test), max_k=args.max_k, device=torch.device(args.device))
    return train_dset, test_dset

class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, features, lengths, n_classes, device=torch.device('cuda')):
        self.features = features.to(device)
        self.lengths = lengths.to(device)
        self.n_classes = n_classes

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        return {
            'features': self.features[index],
            'lengths': self.lengths[index]
        }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, features, lengths, valid_classes, max_k, n_classes=None, device=torch.device('cuda')):
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.lengths = lengths.to(device)
        if valid_classes is not None:
            valid_classes = [c.to(device) for c in valid_classes]
        self.valid_classes = valid_classes
        self.max_k = max_k
        if n_classes is None:
            n_classes = features.size(2)
        self.n_classes = n_classes

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        labels = self.labels[index]
        spans = labels_to_spans(labels.unsqueeze(0), max_k=self.max_k).squeeze(0)
        if self.valid_classes is None:
            return {
                'labels': self.labels[index],
                'features': self.features[index],
                'lengths': self.lengths[index],
                'spans': spans
            }
        else:
            return {
                'labels': self.labels[index],
                'features': self.features[index],
                'lengths': self.lengths[index],
                'valid_classes': self.valid_classes[index],
                'spans': spans
            }

def make_features(labels, n_classes, shift_constant=1.0):
    batch_size_, N_ = labels.size()
    f = torch.randn((batch_size_, N_, n_classes)).to(labels.device)
    shift = torch.zeros_like(f)
    shift.scatter_(2, labels.unsqueeze(2), shift_constant)
    return shift + f

def synthetic_data(num_points=200, n_classes=3, max_seq_len=20, K=5, classes_per_seq=None):
    labels = []
    lengths = []
    valid_classes = []
    for i in range(num_points):
        if i == 0:
            length = max_seq_len
        else:
            length = random.randint(K, max_seq_len)
        lengths.append(length)
        seq = []
        current_step = 0
        if classes_per_seq is not None:
            assert classes_per_seq <= n_classes
            valid_classes_ = np.random.choice(list(range(n_classes)), size=classes_per_seq, replace=False)
        else:
            valid_classes_ = list(range(n_classes))
        valid_classes.append(valid_classes_)
        while len(seq) < max_seq_len:
            step_len = random.randint(1, K-1)
            seq_ = valid_classes_[current_step % len(valid_classes_)]
            seq.extend([seq_] * step_len)
            current_step += 1
        seq = seq[:max_seq_len]
        labels.append(seq)
    labels = torch.LongTensor(labels)
    features = make_features(labels, n_classes, shift_constant=1.)
    lengths = torch.LongTensor(lengths)
    valid_classes = [torch.LongTensor(c) for c in valid_classes]
    return labels, features, lengths, valid_classes

def unit_test_data(num_points=200, seq_len=20, n_dim=2):
    labels = []
    features = []
    lengths = []
    for i in range(num_points):
        lengths.append(seq_len)
        seq = []
        while len(seq) < seq_len:
            seq.extend([0] * 4)#random.randint(5, 5))
            seq.extend([1] * 1)#random.randint(1, 1))
        seq = seq[:seq_len]
        feat = np.zeros((seq_len, n_dim))
        for j, label in enumerate(seq):
            if label == 1:
                feat[j, random.randint(0, n_dim - 1)] = 1
        labels.append(seq)
        features.append(feat)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(np.stack(features, axis=0))
    lengths = torch.LongTensor(lengths)
    return labels, features, lengths, None

def nbc_annotations_dataset(mode='train', subsample=45):
    paths = glob.glob(NBC_ROOT + 'tuning/*.txt')
    assert len(paths) > 0

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

    annotations = []
    for path in paths:
        sess = path.replace('\\', '/').split('/')[-1].replace('_annotations.txt', '')
        if mode == 'train':
            if sess[:4] in ['3_1b', '4_2b']:
                continue
        else:
            assert mode == 'test'
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

def nbc_synthetic_dataset(num_points=100):
    min_seq_len = 20
    max_seq_len = 50
    idle_expectation = 10
    action_expectation = 2
    classes = [0, 1, 2, 3] #idle, reach, pick, put

    labels = []
    lengths = []
    valid_classes = []
    for i in range(num_points):
        if i == 0:
            length = max_seq_len
        else:
            length = random.randint(min_seq_len, max_seq_len)
        lengths.append(length)
        seq = []
        valid_classes.append(classes)
        while len(seq) < max_seq_len:
            seq.extend([0] * np.random.poisson(idle_expectation)) #idle
            seq.extend([1] * np.random.poisson(action_expectation)) #reach
            if random.random() < .7:
                seq.extend([2] * np.random.poisson(action_expectation)) #pick
                if random.random() < .7:
                    if random.random() < .7:
                        seq.extend([0] * np.random.poisson(action_expectation)) #gap between pick/put
                    seq.extend([3] * np.random.poisson(action_expectation)) #put
        seq = seq[:max_seq_len]
        labels.append(seq)
    labels = torch.LongTensor(labels)
    features = make_features(labels, len(classes), shift_constant=1.)
    lengths = torch.LongTensor(lengths)
    valid_classes = [torch.LongTensor(c) for c in valid_classes]
    return labels, features, lengths, valid_classes

def rle_spans(spans, lengths):
    b, _ = spans.size()
    all_rle = []
    for i in range(b):
        rle_ = []
        spans_ = spans[i, :lengths[i]]
        symbol_ = None
        count = 0
        for symbol in spans_:
            symbol = symbol.item()
            if symbol_ is None or symbol != -1:
                if symbol_ is not None:
                    assert count > 0
                    rle_.append((symbol_, count))
                count = 0
                symbol_ = symbol
            count += 1
        if symbol_ is not None:
            assert count > 0
            rle_.append((symbol_, count))
        assert sum(count for _, count in rle_) == lengths[i]
        all_rle.append(rle_)
    return all_rle

def labels_to_spans(labels, max_k):
    _, N = labels.size()
    prev = labels[:, 0]
    values = [prev.unsqueeze(1)]
    lengths = torch.ones_like(prev)
    for n in range(1, N):
        label = labels[:, n]
        same_symbol = (prev == label)
        if max_k is not None:
            same_symbol = same_symbol & (lengths < max_k - 1)
        encoded = torch.where(same_symbol, torch.full([1], -1, device=same_symbol.device, dtype=torch.long), label)
        lengths = torch.where(same_symbol, lengths, torch.full([1], 0, device=same_symbol.device, dtype=torch.long))
        lengths += 1
        values.append(encoded.unsqueeze(1))
        prev = label
    return torch.cat(values, dim=1)

def spans_to_labels(spans):
    _, N = spans.size()
    labels = spans[:, 0]
    assert (labels != -1).all()
    values = [labels.unsqueeze(1)]
    for n in range(1, N):
        span = spans[:, n]
        labels_ = torch.where(span == -1, labels, span)
        values.append(labels_.unsqueeze(1))
        labels = labels_
    return torch.cat(values, dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    train_dset, test_dset = dataset_from_args(args)
    print(train_dset[0]['features'])
    print(train_dset[0]['features'].shape)
    print(test_dset[0]['features'].shape)
