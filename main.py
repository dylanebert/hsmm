import numpy as np
import pandas as pd
import torch
import random
import argparse
from hsmm import SemiMarkovModule, optimal_map, spans_to_labels
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import config
import bridge

random.seed(a=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SemiMarkovDataset(torch.utils.data.Dataset):
    def __init__(self, features, steps, lengths):
        n = len(features)
        max_seq_len = max(lengths)
        d = features[0].shape[-1]

        _features = np.zeros((n, max_seq_len, d))
        _lengths = np.zeros((n,))
        _labels = np.zeros((n, max_seq_len))
        _steps = np.zeros((n, max_seq_len))
        for i in range(n):
            feat = features[i]
            steps_ = steps[i]
            length = lengths[i]
            _features[i, :feat.shape[0]] = feat
            _lengths[i] = length
            _steps[i, :feat.shape[0]] = steps_
            _labels[i, :feat.shape[0]] = 1
        print(_features.shape)

        device = torch.device('cuda')
        self.features = torch.FloatTensor(_features).to(device)
        self.lengths = torch.LongTensor(_lengths).to(device)
        self.labels = torch.LongTensor(_labels).to(device)
        self.steps = torch.LongTensor(_steps).to(device)

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

def train_supervised(args, train_dset, dev_dset):
    n_dim = train_dset[0]['features'].shape[-1]
    model = SemiMarkovModule(args, n_dim).cuda()

    features = []
    lengths = []
    labels = []
    for i in range(len(train_dset)):
        sample = train_dset[i]
        features.append(sample['features'])
        lengths.append(sample['lengths'])
        labels.append(sample['labels'])

    model.fit_supervised(features, labels, lengths)
    if args.debug:
        debug(model, dev_dset, 0)
    return model

def train_unsupervised(args, train_dset, dev_dset):
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)
    dev_loader = torch.utils.data.DataLoader(dev_dset, batch_size=10)

    n_dim = train_dset[0]['features'].shape[-1]
    model = SemiMarkovModule(args, n_dim).cuda()
    model.initialize_gaussian(train_dset.features, train_dset.lengths)
    optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

    if len(args.overrides) > 0:
        features = []
        lengths = []
        labels = []
        for i in range(len(train_dset)):
            sample = train_dset[i]
            features.append(sample['features'])
            lengths.append(sample['lengths'])
            labels.append(sample['labels'])
        model.initialize_supervised(features, labels, lengths, overrides=args.overrides, freeze=True)

    model.train()
    best_loss = 1e9
    best_model = SemiMarkovModule(args, n_dim).cuda()
    k = 0; patience = 5

    def report_acc():
        gold, pred, pred_remapped = predict(model, dev_dset)
        eval(gold, pred_remapped)

    epoch = 0
    while True:
        losses = []
        for batch in train_loader:
            features = batch['features']
            lengths = batch['lengths']
            N_ = lengths.max().item()
            features = features[:, :N_, :]
            loss = model.log_likelihood(features, lengths, None)
            loss = -loss
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            model.zero_grad()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, np.mean(losses)))
        report_acc()
        if args.debug:
            debug(model, dev_dset, epoch)
        epoch += 1
        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses) - 1e-3
            best_model.load_state_dict(model.state_dict())
            k = 0
        else:
            k += 1
            print('Loss didn\'t improve for {} epochs'.format(k))
            if k == patience:
                print('Stopping')
                break

    return best_model

def predict(model, dset):
    data = torch.utils.data.DataLoader(dset, batch_size=10)

    gold = []; pred = []; pred_remapped = []
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

        valid_classes = torch.LongTensor(np.array(list(range(model.n_classes)), dtype=int)).to(torch.device('cuda'))
        for i in range(batch_size):
            pred_remapped_, mapping = optimal_map(pred_labels_trim[i], gold_labels_trim[i], valid_classes)
            pred_remapped += [pred_remapped_.cpu().numpy()]

        gold += [a.cpu().numpy() for a in gold_labels_trim]
        pred += [a.cpu().numpy() for a in pred_labels_trim]

    return gold, pred, pred_remapped

def eval(gold, pred):
    match, total = 0, 0
    a_match, a_total = 0, 0
    for gold_, pred_ in zip(gold, pred):
        match += (gold_ == pred_).sum()
        a_match += (gold_ == pred_)[gold_ > 0].sum()
        total += gold_.shape[0]
        a_total += gold_[gold_ > 0].shape[0]
    print('Accuracy: {:.2f}, Action Accuracy: {:.2f}'.format(100. * match / total, 100. * a_match / a_total))

def debug(model, dev_dset, epoch):
    features = dev_dset[0]['features'].unsqueeze(0)
    labels = dev_dset[0]['labels'].unsqueeze(0)
    lengths = dev_dset[0]['lengths'].unsqueeze(0)

    params = {
        'features': features.cpu().numpy(),
        'labels': labels.cpu().numpy(),
        'trans': np.exp(model.transition_log_probs(None).detach().cpu().numpy()),
        'emission': np.exp(model.emission_log_probs(features, None).detach().cpu().numpy()),
        'initial': np.exp(model.initial_log_probs(None).detach().cpu().numpy()),
        'lengths': np.exp(model.poisson_log_rates.detach().cpu().numpy()),
        'mean': model.gaussian_means.detach().cpu().numpy(),
        'cov': model.gaussian_cov.cpu().numpy()
    }

    np.set_printoptions(suppress=True)
    for param in ['features', 'labels', 'mean', 'cov', 'trans', 'lengths']:
        print('{}\n{}\n'.format(param, params[param]))

def viz(gold, pred):
    gold = gold[np.newaxis, :]
    pred = pred[np.newaxis, :]
    N = gold.shape[-1]

    print(gold)
    print(pred)

    gold = np.tile(gold, (N // 10, 1))
    pred = np.tile(pred, (N // 10, 1))

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
    class HSMMArgs:
        def __init__(self):
            self.sm_allow_self_transitions = True
            self.sm_lr = 1e-1
            self.sm_supervised_state_smoothing = 1e-2
            self.sm_supervised_length_smoothing = 1e-1
            self.sm_supervised_cov_smoothing = 0.
            self.supervised = False
            self.max_k = 5
            self.overrides = []
            self.debug = True
            self.n_classes = 5
    hsmm_args = HSMMArgs()
    data_args = config.deserialize('vae8')
    sequences = bridge.get_hsmm_sequences(data_args)
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()
    data = {}
    for type in ['train', 'dev', 'test']:
        feat, steps, lengths = sequences[type]
        data[type] = SemiMarkovDataset(feat, steps, lengths)

    if hsmm_args.supervised:
        model = train_supervised(hsmm_args, data['train'], data['dev'])
    else:
        model = train_unsupervised(hsmm_args, data['train'], data['dev'])
    gold, pred, pred_remapped = predict(model, data['train'])
    eval(gold, pred)
    eval(gold, pred_remapped)
    viz(gold[0], pred[0])
