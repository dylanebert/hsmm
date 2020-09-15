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

class ValidateFeatures(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        valid_args = ['obj_speeds', 'lhand_speed', 'rhand_speed']
        for value in values:
            if value not in valid_args:
                raise ValueError(value)
        setattr(args, self.dest, values)

def add_args(parser):
    parser.add_argument('--nbc_features', nargs='+', action=ValidateFeatures, default=['obj_speeds'])
    parser.add_argument('--nbc_subsample', type=int, default=90)
    parser.add_argument('--nbc_mode', choices=['train', 'test'], default='train')

class NBCData:
    def __init__(self, args):
        self.args = args
        self.load_spatial()
        self.load_features()

    def load_spatial(self):
        print('Loading spatial data')
        paths = glob.glob(NBC_ROOT + 'raw/*')
        assert len(paths) > 0
        spatial = []
        for path in tqdm(paths):
            session = path.replace('\\', '/').split('/')[-1]
            if self.args.nbc_mode == 'train':
                if session[:4] in ['3_1b', '4_2b']:
                    continue
            else:
                assert self.args.nbc_mode == 'test'
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

    def subsample(self, session):
        step = self.args.subsample
        spatial = pd.read_json(os.path.join(session, 'spatial.json'), orient='index')
        start, end = int(spatial.iloc[0]['step']), int(spatial.iloc[-1]['step'])
        steps = list(range(start, end, step))
        spatial = spatial[spatial['step'].isin(steps)]
        spatial.to_json(os.path.join(session, 'spatial_subsample{}.json'.format(step)), orient='index')
        return spatial

    def load_features(self):
        if 'obj_speeds' in self.args.nbc_features:
            features, lengths = self.obj_speeds()
            print(features.shape)
            print(lengths)
            print(lengths.shape)

    def obj_speeds(self):
        df = self.spatial[(self.spatial['dynamic'] == True) & ~(self.spatial['name'].isin(['LeftHand', 'RightHand', 'Head']))]
        df['speed'] = df.apply(lambda row: np.linalg.norm([row['velX'], row['velY'], row['velZ']]), axis=1)
        names = {k: v for k, v in enumerate(sorted(df['name'].unique()))}
        feat = []
        k = 0
        for session, group in df.groupby('session'):
            seq = []
            for step, rows in group.groupby('step'):
                vec = np.zeros((len(names),), dtype=np.float32)
                for idx, obj in names.items():
                    obj_row = rows[rows['name'] == obj]
                    if not len(obj_row) == 1:
                        continue
                    obj_row = obj_row.iloc[0]
                    vec[idx] = obj_row['speed']
                seq.append(vec)
            if len(seq) > k:
                k = len(seq)
            seq = np.vstack(seq)
            feat.append(seq)
        padded = []
        lengths = []
        for vec in feat:
            vec_ = np.zeros((k, len(names)), dtype=np.float32)
            vec_[:vec.shape[0]] = vec
            padded.append(vec_)
            lengths.append(vec.shape[0])
        return np.stack(padded, axis=0), np.array(lengths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    data = NBCData(args)
