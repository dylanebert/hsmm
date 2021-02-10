import numpy as np
import pandas as pd
import torch
import random
import argparse
from hsmm import SemiMarkovModule, optimal_map, spans_to_labels
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
from sklearn import preprocessing
import json

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
import config

class SemiMarkovDataset(torch.utils.data.Dataset):
    def __init__(self, features, lengths):
        n = len(features)
        max_seq_len = max(lengths)
        d = features[0].shape[-1]

        _features = np.zeros((n, max_seq_len, d))
        _lengths = np.zeros((n,))
        _labels = np.zeros((n, max_seq_len))
        for i in range(n):
            feat = features[i]
            length = lengths[i]
            _features[i, :feat.shape[0]] = feat
            _lengths[i] = length
            _labels[i, :feat.shape[0]] = 1
        print(_features.shape)

        device = torch.device('cuda')
        self.features = torch.FloatTensor(_features).to(device)
        self.lengths = torch.LongTensor(_lengths).to(device)
        self.labels = torch.LongTensor(_labels).to(device)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        batch = {
            'features': self.features[index],
            'lengths': self.lengths[index],
            'labels': self.labels[index]
        }
        return batch

def viz(pred):
    pred = pred[np.newaxis, :]
    N = pred.shape[-1]
    print(pred)
    pred = np.tile(pred, (N // 10, 1))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(pred)
    ax1.axis('off')
    plt.tight_layout()
    plt.show()

class HSMMWrapper:
    def __init__(self, args, nbc_wrapper, autoencoder_wrapper):
        self.args = args
        self.nbc_wrapper = nbc_wrapper
        self.autoencoder_wrapper = autoencoder_wrapper
        self.prepare_data()
        self.get_hsmm()

    def get_hsmm(self):
        if self.try_load_cached():
            return
        self.train()
        self.predictions = {}
        for type in ['train', 'dev', 'test']:
            self.predictions[type] = self.predict(type)
        self.cache()

    def try_load_cached(self, load_model=False):
        savefile = config.find_savefile(self.args, 'hsmm')
        if savefile is None:
            return False
        weights_path = NBC_ROOT + 'tmp/hsmm/{}_weights.pt'.format(savefile)
        predictions_path = NBC_ROOT + 'tmp/hsmm/{}_predictions.json'.format(savefile)
        if load_model:
            self.model = SemiMarkovModule(self.args, self.n_dim).cuda()
            self.model.load_state_dict(torch.load(weights_path))
        with open(predictions_path) as f:
            self.predictions = json.load(f)
        print('loaded cached hsmm')
        return True

    def cache(self):
        savefile = config.generate_savefile(self.args, 'hsmm')
        weights_path = NBC_ROOT + 'tmp/hsmm/{}_weights.pt'.format(savefile)
        predictions_path = NBC_ROOT + 'tmp/hsmm/{}_predictions.json'.format(savefile)
        torch.save(self.model.state_dict(), weights_path)
        with open(predictions_path, 'w+') as f:
            json.dump(self.predictions, f)
        print('cached hsmm')

    def prepare_data(self):
        def aggregate_sessions(z, steps):
            sessions = {}
            for i, (key, steps_) in enumerate(steps):
                session = key[0]
                feat = z[i]
                if session not in sessions:
                    sessions[session] = {'feat': [], 'indices': []}
                sessions[session]['feat'].append(feat)
                sessions[session]['indices'].append(i)
            features, lengths, indices = [], [], []
            for session in sessions.keys():
                feat = np.array(sessions[session]['feat'], dtype=np.float32)
                indices_ = np.array(sessions[session]['indices'], dtype=int)
                features.append(feat)
                lengths.append(feat.shape[0])
                indices.append(indices_)
            return features, lengths, indices

        def preprocess(sequences):
            scaler = preprocessing.StandardScaler().fit(np.vstack(sequences['train'][0]))
            for type in ['train', 'dev', 'test']:
                feat, lengths, indices = sequences[type]
                feat = scaler.transform(np.vstack(feat))
                feat_ = []
                i = 0
                for length in lengths:
                    feat_.append(feat[i:i+length,:])
                    i += length
                assert i == feat.shape[0]
                sequences[type] = (feat_, lengths, indices)
            return sequences

        sequences = {}
        lengths = {}
        for type in ['train', 'dev', 'test']:
            z = self.autoencoder_wrapper.encodings[type]
            steps = list(self.nbc_wrapper.nbc.steps[type].items())
            feat, lengths, indices = aggregate_sessions(z, steps)
            sequences[type] = (feat, lengths, indices)
        self.sequences = preprocess(sequences)

        self.data = {}
        for type in ['train', 'dev', 'test']:
            feat, lengths, indices = self.sequences[type]
            self.data[type] = SemiMarkovDataset(feat, lengths)
        self.n_dim = self.data['train'][0]['features'].shape[-1]

    def train(self):
        if self.args.sm_supervised:
            self.train_supervised()
        else:
            self.train_unsupervised()

    def train_supervised(self):
        self.model = SemiMarkovModule(self.args, self.n_dim).cuda()
        features = []
        lengths = []
        labels = []
        for i in range(len(self.data['train'])):
            sample = self.data['train'][i]
            features.append(sample['features'])
            lengths.append(sample['lengths'])
            labels.append(sample['labels'])

        self.model.fit_supervised(features, labels, lengths)
        if self.args.sm_debug:
            self.debug()

    def train_unsupervised(self):
        train_loader = torch.utils.data.DataLoader(self.data['train'], batch_size=10)
        dev_loader = torch.utils.data.DataLoader(self.data['dev'], batch_size=10)

        self.model = SemiMarkovModule(self.args, self.n_dim).cuda()
        self.model.initialize_gaussian(self.data['train'].features, self.data['train'].lengths)
        optimizer = torch.optim.Adam(self.model.parameters(), self.model.learning_rate)

        if len(self.args.sm_overrides) > 0:
            features = []
            lengths = []
            labels = []
            for i in range(len(self.data['train'])):
                sample = self.data['train'][i]
                features.append(sample['features'])
                lengths.append(sample['lengths'])
                labels.append(sample['labels'])
            self.model.initialize_supervised(features, labels, lengths, overrides=self.args.sm_overrides, freeze=True)

        best_loss = 1e9
        best_model = SemiMarkovModule(self.args, self.n_dim).cuda()
        k = 0; patience = 5
        epoch = 0
        self.model.train()
        while True:
            losses = []
            for batch in train_loader:
                features = batch['features']
                lengths = batch['lengths']
                N_ = lengths.max().item()
                features = features[:, :N_, :]
                loss = self.model.log_likelihood(features, lengths, None)
                loss = -loss
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                self.model.zero_grad()
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, np.mean(losses)))
            if self.args.sm_debug:
                self.debug()
            epoch += 1
            if np.mean(losses) < best_loss:
                best_loss = np.mean(losses) - 1e-3
                best_model.load_state_dict(self.model.state_dict())
                k = 0
            else:
                k += 1
                print('Loss didn\'t improve for {} epochs'.format(k))
                if k == patience:
                    print('Stopping')
                    break

        self.model = best_model

    def predict(self, type='dev'):
        data = torch.utils.data.DataLoader(self.data[type], batch_size=10)
        pred = []
        for batch in data:
            features = batch['features']
            lengths = batch['lengths']
            batch_size = features.size(0)
            N_ = lengths.max().item()
            features = features[:, :N_, :]
            pred_spans = self.model.viterbi(features, lengths, valid_classes_per_instance=None, add_eos=True)
            pred_labels = spans_to_labels(pred_spans)
            pred_labels_trim = self.model.trim(pred_labels, lengths)
            pred += [x.cpu().numpy().tolist() for x in pred_labels_trim]
        return pred

    def debug(self, type='dev'):
        features = self.data['dev'][0]['features'].unsqueeze(0)
        labels = self.data['dev'][0]['labels'].unsqueeze(0)
        lengths = self.data['dev'][0]['lengths'].unsqueeze(0)

        params = {
            'features': features.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'trans': np.exp(self.model.transition_log_probs(None).detach().cpu().numpy()),
            'emission': np.exp(self.model.emission_log_probs(features, None).detach().cpu().numpy()),
            'initial': np.exp(self.model.initial_log_probs(None).detach().cpu().numpy()),
            'lengths': np.exp(self.model.poisson_log_rates.detach().cpu().numpy()),
            'mean': self.model.gaussian_means.detach().cpu().numpy(),
            'cov': self.model.gaussian_cov.cpu().numpy()
        }

        np.set_printoptions(suppress=True)
        for param in ['mean', 'cov', 'trans', 'lengths']:
            print('{}\n{}\n'.format(param, params[param]))
