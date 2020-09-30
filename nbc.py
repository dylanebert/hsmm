import numpy as np
import pandas as pd
import argparse
import sys
import os
import glob
from tqdm import tqdm
import torch

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT envvar'
NBC_ROOT = os.environ['NBC_ROOT']
assert os.path.exists(NBC_ROOT), 'NBC_ROOT location doesn\'t exist'

def add_args(parser):
    parser.add_argument('--nbc_features', nargs='+', choices=['obj_speeds', 'lhand_speed', 'rhand_speed', 'apple_speed'], default=['obj_speeds'])
    parser.add_argument('--nbc_labels', choices=['moving'], default='moving')
    parser.add_argument('--nbc_subsample', type=int, default=90)
    parser.add_argument('--nbc_mini', action='store_true')
    parser.add_argument('--nbc_filter', help='filter out all-zero features', action='store_true')
    parser.add_argument('--nbc_filter_window', type=int, default=5)

class NBCData:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.load_spatial()
        self.generate_features()
        self.generate_labels()
        if args.nbc_filter:
            self.filter_zeros()
        self.collate()

    def to_dataset(self):
        if self.labels is None:
            labels = None
        else:
            labels = torch.LongTensor(self.labels)
        features = torch.FloatTensor(self.features)
        lengths = torch.LongTensor(self.lengths)
        valid_classes = None
        return labels, features, lengths, valid_classes

    def collate(self):
        max_seq_len = 0
        _, d = self.features[0].shape
        n = len(self.features)
        for feat, lbls in zip(self.features, self.labels):
            assert len(feat) == len(lbls)
            if len(feat) > max_seq_len:
                max_seq_len = len(feat)
            _, d = feat.shape

        features = np.zeros((n, max_seq_len, d))
        labels = np.zeros((n, max_seq_len))
        lengths = np.zeros((n,))
        for i, (feat, lbls) in enumerate(zip(self.features, self.labels)):
            lengths[i] = len(feat)
            features[i, :len(feat), :] = feat
            labels[i, :len(lbls)] = lbls

        self.features = features
        self.labels = labels
        self.lengths = lengths

    def load_spatial(self):
        paths = glob.glob(NBC_ROOT + 'raw/*')
        assert len(paths) > 0
        spatial = []
        for path in paths:
            session = path.replace('\\', '/').split('/')[-1]
            if self.mode == 'train':
                if self.args.nbc_mini and not session[:4] == '1_1a':
                    continue
                if session[:4] in ['3_1b', '4_2b']:
                    continue
            else:
                assert self.mode == 'test'
                if self.args.nbc_mini and not session[:4] == '3_1b':
                    continue
                if session[:4] not in ['3_1b', '4_2b']:
                    continue
            subsample_path = os.path.join(path, 'spatial_subsample{}.json'.format(self.args.nbc_subsample))
            if os.path.exists(subsample_path):
                df = pd.read_json(subsample_path, orient='index')
            else:
                print('Couldn\'t find {} - creating subsampled file'.format(subsample_path))
                df = self.subsample(path)
            df['session'] = session
            spatial.append(df)
        self.spatial = pd.concat(spatial)

        self.spatial['speed'] = self.spatial.apply(lambda row: np.linalg.norm([row['velX'], row['velY'], row['velZ']]), axis=1)
        self.spatial.fillna(0, inplace=True)
        self.spatial['speed'].clip(0., 3., inplace=True)

        self.sessions = sorted(self.spatial['session'].unique())

    def subsample(self, session):
        step = self.args.subsample
        spatial = pd.read_json(os.path.join(session, 'spatial.json'), orient='index')
        start, end = int(spatial.iloc[0]['step']), int(spatial.iloc[-1]['step'])
        steps = list(range(start, end, step))
        spatial = spatial[spatial['step'].isin(steps)]
        spatial.to_json(os.path.join(session, 'spatial_subsample{}.json'.format(step)), orient='index')
        return spatial

    def generate_features(self):
        self.features = []
        for session in self.sessions:
            features = []
            if 'obj_speeds' in self.args.nbc_features:
                for obj in sorted(self.spatial[(self.spatial['dynamic'] == True) & ~(self.spatial['name'].isin(['LeftHand', 'RightHand', 'Head']))]['name'].unique()):
                    feat = self.obj_speed(session, obj)
                    features.append(feat)
            if 'lhand_speed' in self.args.nbc_features:
                feat = self.obj_speed(session, 'LeftHand')
                features.append(feat)
            if 'rhand_speed' in self.args.nbc_features:
                feat = self.obj_speed(session, 'RightHand')
                features.append(feat)
            if 'apple_speed' in self.args.nbc_features:
                feat = self.obj_speed(session, 'Apple')
                features.append(feat)
            features = np.stack(features, axis=-1)
            self.features.append(features)

    def generate_labels(self):
        labels = []
        assert self.args.nbc_labels == 'moving'
        for feat in self.features:
            labels_ = np.any(feat > 0, axis=-1).astype(int)
            labels.append(labels_)
        self.labels = labels

    def filter_zeros(self):
        features = []
        labels = []
        for feat, lbls in zip(self.features, self.labels):
            rolled = np.roll(feat, self.args.nbc_filter_window, axis=0) + feat + np.roll(feat, -self.args.nbc_filter_window, axis=0)
            zero_mask = np.all(rolled == 0, axis=-1)
            if np.all(zero_mask == True):
                continue
            feat = feat[~zero_mask]
            lbls = lbls[~zero_mask]
            features.append(feat)
            labels.append(lbls)
        self.features = features
        self.labels = labels

    def obj_speed(self, session, obj):
        group = self.spatial[self.spatial['session'] == session]
        n = len(group['step'].unique())
        seq = np.zeros((n,))
        for i, (step, rows) in enumerate(group.groupby('step')):
            row = rows[rows['name'] == obj]
            if not len(row) == 1:
                #print('Speed not found for {} at step {}'.format(obj, step))
                continue
            row = row.iloc[0]
            seq[i] = row['speed']
        return seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    data = NBCData(args)
