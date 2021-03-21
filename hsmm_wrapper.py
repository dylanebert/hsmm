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
from input_modules import InputModule

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)

class SemiMarkovDataset(torch.utils.data.Dataset):
    def __init__(self, features, lengths, device):
        self.features = torch.FloatTensor(features).to(device)
        self.lengths = torch.LongTensor(lengths).to(device)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        batch = {
            'features': self.features[index],
            'lengths': self.lengths[index]
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
    def __init__(self, fname, device='cpu'):
        if '/' in fname or '\\' in fname:
            fname = os.path.basename(fname)
        fname = fname.replace('.json', '')
        self.fname = fname
        with open(NBC_ROOT + 'config/{}.json'.format(fname)) as f:
            args = json.load(f)
        self.args = args
        self.input_module = InputModule(args['input_config'])
        self.device = torch.device(device)
        if self.load():
            return
        self.data = {}
        self.n_dim = self.input_module.z['train'].shape[-1]
        for type in ['train', 'dev', 'test']:
            self.data[type] = SemiMarkovDataset(self.input_module.z[type], self.input_module.lengths[type], self.device)
        self.train_unsupervised()
        self.save()

    def save(self):
        self.predictions = {}
        for type in ['train', 'dev', 'test']:
            self.predictions[type] = self.predict(type)
        weights_path = NBC_ROOT + 'cache/hsmm/{}_weights.json'.format(self.fname)
        predictions_path = NBC_ROOT + 'cache/hsmm/{}_predictions.json'.format(self.fname)
        torch.save(self.model.state_dict(), weights_path)
        with open(predictions_path, 'w+') as f:
            json.dump(self.predictions, f)
        print('saved to {}'.format(predictions_path))

    def load(self, load_model=False):
        weights_path = NBC_ROOT + 'cache/hsmm/{}_weights.json'.format(self.fname)
        predictions_path = NBC_ROOT + 'cache/hsmm/{}_predictions.json'.format(self.fname)
        if not os.path.exists(weights_path) or not os.path.exists(predictions_path):
            return False
        if load_model:
            assert False
        with open(predictions_path) as f:
            self.predictions = json.load(f)
        print('loaded from {}'.format(predictions_path))
        return True

    def train_unsupervised(self):
        train_loader = torch.utils.data.DataLoader(self.data['train'], batch_size=10)
        dev_loader = torch.utils.data.DataLoader(self.data['dev'], batch_size=10)

        self.model = SemiMarkovModule(self.n_dim, self.args['n_classes'], self.args['max_k'], self.args['allow_self_transitions'], self.args['cov_factor']).to(self.device)
        self.model.initialize_gaussian(self.data['train'].features, self.data['train'].lengths)
        optimizer = torch.optim.Adam(self.model.parameters(), self.model.learning_rate)

        best_loss = 1e9
        best_model = SemiMarkovModule(self.n_dim, self.args['n_classes'], self.args['max_k'], self.args['allow_self_transitions'], self.args['cov_factor']).to(self.device)
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
        lengths = self.data['dev'][0]['lengths'].unsqueeze(0)

        params = {
            'features': features.cpu().numpy(),
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

class Args:
    def __init__(self):
        return

if __name__ == '__main__':
    HSMMWrapper('hsmm_default', device='cuda')
