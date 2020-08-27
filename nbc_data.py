import numpy as np
import pandas as pd
import glob
import torch
import h5py
import subprocess
from tqdm import tqdm

def make_features(class_labels, n_classes, shift_constant=1.0):
    batch_size_, N_ = class_labels.size()
    f = torch.randn((batch_size_, N_, n_classes))
    shift = torch.zeros_like(f)
    shift.scatter_(2, class_labels.unsqueeze(2), shift_constant)
    return shift + f

def annotations_to_dataset(mode='train'):
    paths = glob.glob('D:/nbc/tuning/*.txt')
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
        spatial_ = pd.read_json('D:/nbc/raw/{}/spatial.json'.format(sess), orient='index')
        start, end = spatial_.iloc[0]['step'], spatial_.iloc[-1]['step']
        annotations.append((annotations_, start, end))

    mapping = {'idle': 0}
    N = 0
    for a, start, end in annotations:
        if end - start > N:
            N = end - start
        for action in a['action'].values:
            if action not in mapping:
                mapping[action] = len(mapping)

    labels = []
    lengths = []
    valid_classes = []
    for a, start, end in annotations:
        lengths.append(end - start)
        seq = []
        valid_classes_ = []
        for i in range(start, end):
            label = 0
            for _, row in a.iterrows():
                if i in range(row['start_step'], row['end_step']):
                    label = mapping[row['action']]
            seq.append(label)
            if label not in valid_classes_:
                valid_classes_.append(label)
        while len(seq) < N:
            seq.append(0)
        labels.append(seq)
        valid_classes.append(valid_classes_)
    features = make_features(torch.LongTensor(labels), len(mapping)).detach().cpu().numpy()

    with h5py.File('data\\nbc_toy_{}.h5'.format(mode), 'w') as f:
        f.create_dataset('labels', data=np.array(labels, dtype=int))
        f.create_dataset('features', data=np.array(features, dtype=np.float32))
        f.create_dataset('lengths', data=np.array(lengths, dtype=int))

def get_onehot_dataset(mode='train', subsample=90):
    with h5py.File('data\\nbc_toy_{}.h5'.format(mode), 'r') as f:
        labels = np.array(f['labels'])
        features = np.array(f['features'])
        lengths = np.array(f['lengths'])
    indices = list(range(0, labels.shape[1], subsample))
    labels = labels[:, indices]
    features = features[:, indices, :]
    lengths = lengths // subsample
    labels = torch.LongTensor(labels)
    features = torch.Tensor(features)
    lengths = torch.LongTensor(lengths)
    return labels, features, lengths, None

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
    #annotations_to_dataset('test')
    labels, features, lengths, valid_classes = get_onehot_dataset('train')
    max_k = compute_max_k(labels)
    baseline = compute_baseline(labels)
    print(baseline)

    #annotations_to_dataset('train')
