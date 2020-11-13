import numpy as np
import pandas as pd
import torch
import random
import argparse
from hsmm import SemiMarkovModule, optimal_map, spans_to_labels
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append('C:/Users/dylan/Documents/')
from nbc.nbc import NBC

random.seed(a=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SemiMarkovDataset(torch.utils.data.Dataset):
    def __init__(self, args, dset, type='train'):
        n = len(dset.features[type])
        max_seq_len = 0
        for feat in dset.features[type].values():
            if len(feat) > max_seq_len:
                max_seq_len = len(feat)
            d = feat.shape[-1]

        features = np.zeros((n, max_seq_len, d))
        lengths = np.zeros((n,))
        labels = np.zeros((n, max_seq_len))
        steps = np.zeros((n, max_seq_len))
        for i, session in enumerate(dset.features[type].keys()):
            feat = np.abs(dset.features[type][session])
            steps_ = dset.sequences[type][session]['step'].unique()
            features[i, :feat.shape[0]] = dset.features[type][session]
            lengths[i] = feat.shape[0]
            labels[i, :feat.shape[0]] = np.any(feat > 1e-1, axis=1).astype(np.int)
            steps[i, :feat.shape[0]] = steps_

        device = torch.device(args.device)
        self.features = torch.FloatTensor(features).to(device)
        self.lengths = torch.LongTensor(lengths).to(device)
        self.labels = torch.LongTensor(labels).to(device)
        self.steps = torch.LongTensor(steps).to(device)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        batch = {
            'features': self.features[index],
            'lengths': self.lengths[index],
            'labels': self.labels[index],
            'steps': self.steps[index]
        }
        return batch

def train_supervised(args, dset):
    n_classes = 2
    n_dim = dset[0]['features'].shape[-1]

    model = SemiMarkovModule(args, n_classes, n_dim)
    if args.device == 'cuda':
        model.cuda()

    features = []
    lengths = []
    labels = []
    for i in range(len(dset)):
        sample = dset[i]
        features.append(sample['features'])
        lengths.append(sample['lengths'])
        labels.append(sample['labels'])

    model.fit_supervised(features, labels, lengths)
    return model

def predict(model, args, dset):
    data = torch.utils.data.DataLoader(dset, batch_size=32)
    device = torch.device(args.device)

    items = []
    token_match, token_total = 0, 0
    for batch in data:
        features = batch['features']
        lengths = batch['lengths']
        gold_labels = batch['labels']

        batch_size = features.size(0)
        N_ = lengths.max().item()
        features = features[:, :N_, :]
        gold_labels = gold_labels[:, :N_]

        pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=None, add_eos=True)
        pred_labels = spans_to_labels(pred_spans)

        gold_labels_trim = model.trim(gold_labels, lengths)
        pred_labels_trim = model.trim(pred_labels, lengths)

        for i in range(batch_size):
            valid_classes_ = torch.LongTensor(np.array(list(range(model.n_classes)), dtype=int)).to(device)
            item = {
                'length': lengths[i].item(),
                'gold_labels': gold_labels[i],
                'pred_labels': pred_labels[i],
                'gold_labels_trim': gold_labels_trim[i],
                'pred_labels_trim': pred_labels_trim[i]
            }
            items.append(item)
            token_match += (gold_labels_trim[i] == pred_labels_trim[i]).sum().item()
            token_total += pred_labels_trim[i].size(0)
    accuracy = 100. * token_match / token_total
    print(accuracy)

def viz(model, item):
    features = item['features'].unsqueeze(0)
    lengths = item['lengths'].unsqueeze(0)
    gold_labels = item['labels'].unsqueeze(0)

    N_ = lengths.max().item()
    features = features[:, :N_, :]

    pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=None, add_eos=True)
    pred_labels = spans_to_labels(pred_spans)

    gold_labels_trim = model.trim(gold_labels, lengths)
    pred_labels_trim = model.trim(pred_labels, lengths)

    gold = gold_labels_trim[0].cpu().numpy()[np.newaxis, :]
    pred = pred_labels_trim[0].cpu().numpy()[np.newaxis, :]

    gold = np.tile(gold, (N_ // 10, 1))
    pred = np.tile(pred, (N_ // 10, 1))

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.imshow(gold)
    ax1.axis('off')

    ax2 = fig.add_subplot(212)
    ax2.imshow(pred)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    NBC.add_args(parser)
    SemiMarkovModule.add_args(parser)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max_k', help='maximum length of one state', type=int, default=10)
    args = parser.parse_args([
        '--subsample', '9',
        '--train_sequencing', 'session',
        '--test_sequencing', 'session',
        '--features',
        'velY:Apple'
    ])

    dset = NBC(args)
    train_dset = SemiMarkovDataset(args, dset, type='train')
    test_dset = SemiMarkovDataset(args, dset, type='test')

    model = train_supervised(args, train_dset)
    predict(model, args, test_dset)
    viz(model, train_dset[0])
