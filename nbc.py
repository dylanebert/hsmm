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
    parser.add_argument('--nbc_labels', choices=['moving'], default=['moving'])
    parser.add_argument('--nbc_subsample', type=int, default=90)
    parser.add_argument('--nbc_mini', action='store_true')

class NBCData:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.load_spatial()
        self.generate_features()
        self.generate_labels()

    def to_dataset(self):
        if self.labels is None:
            labels = None
        else:
            labels = torch.LongTensor(self.labels)
        features = torch.FloatTensor(self.features)
        lengths = torch.LongTensor(self.lengths)
        valid_classes = None
        return labels, features, lengths, valid_classes

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

        self.lengths_dict = {}
        for session, group in self.spatial.groupby('session'):
            self.lengths_dict[session] = len(group['step'].unique())
        self.max_seq_len = max(self.lengths_dict.values())
        self.lengths = np.array(list(self.lengths_dict.values()), dtype=int)

    def subsample(self, session):
        step = self.args.subsample
        spatial = pd.read_json(os.path.join(session, 'spatial.json'), orient='index')
        start, end = int(spatial.iloc[0]['step']), int(spatial.iloc[-1]['step'])
        steps = list(range(start, end, step))
        spatial = spatial[spatial['step'].isin(steps)]
        spatial.to_json(os.path.join(session, 'spatial_subsample{}.json'.format(step)), orient='index')
        return spatial

    def generate_features(self):
        features = []
        if 'obj_speeds' in self.args.nbc_features:
            for obj in sorted(self.spatial[(self.spatial['dynamic'] == True) & ~(self.spatial['name'].isin(['LeftHand', 'RightHand', 'Head']))]['name'].unique()):
                feat = self.obj_speed(obj)
                features.append(feat)
        if 'lhand_speed' in self.args.nbc_features:
            feat = self.obj_speed('LeftHand')
            features.append(feat)
        if 'rhand_speed' in self.args.nbc_features:
            feat = self.obj_speed('RightHand')
            features.append(feat)
        if 'apple_speed' in self.args.nbc_features:
            feat = self.obj_speed('Apple')
            features.append(feat)
        self.features = np.concatenate(features, axis=-1)

    def generate_labels(self):
        if self.args.nbc_labels == 'moving':
            self.labels = np.any(self.features > 0, axis=-1).astype(int)
        else:
            assert self.args.nbc_labels == 'none'

    def obj_speed(self, obj):
        feat = []
        for session in self.lengths_dict.keys():
            group = self.spatial[self.spatial['session'] == session]
            seq = np.zeros((self.max_seq_len,))
            for i, (step, rows) in enumerate(group.groupby('step')):
                row = rows[rows['name'] == obj]
                if not len(row) == 1:
                    #print('Speed not found for {} at step {}'.format(obj, step))
                    continue
                row = row.iloc[0]
                seq[i] = row['speed']
            feat.append(seq[:, np.newaxis])
        return np.stack(feat, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    data = NBCData(args)
