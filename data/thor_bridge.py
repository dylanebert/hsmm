from sklearn.preprocessing import StandardScaler
# import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None
# from scipy.spatial.transform import Rotation as R


assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
assert 'THOR_ROOT' in os.environ, 'set THOR_ROOT'
NBC_ROOT = os.environ['NBC_ROOT']
THOR_ROOT = os.environ['THOR_ROOT']


def load_thor_data():
    savepath = NBC_ROOT + '/cache/thor_data.p'
    if os.path.exists(savepath):
        return pd.read_pickle(savepath)

    dir = THOR_ROOT + '/data/'
    fnames = os.listdir(dir)
    fnames_split = {
        'train': fnames[:-20],
        'dev': fnames[-20:-10],
        'test': fnames[-10:]
    }

    data = []
    for type in ['train', 'dev', 'test']:
        for fname in fnames_split[type]:
            fpath = os.path.join(dir, fname)
            df = pd.read_json(fpath, orient='index').sort_index()
            pos = df[['posX', 'posY', 'posZ']]
            pos.loc[:, 'type'] = type
            pos.loc[:, 'session'] = fname.replace('.json', '')
            pos.loc[:, 'step'] = df.index.values
            # rot_euler = df[['rotX', 'rotY', 'rotZ']].to_numpy()
            # rot = R.from_euler('xyz', rot_euler, degrees=True).as_quat()
            data.append(pos)
    data = pd.concat(data).reset_index(drop=True)
    data.to_pickle(savepath)
    return data


def preprocess_thor_data(data):
    savepath = NBC_ROOT + '/cache/thor_data_preprocessed.p'
    if os.path.exists(savepath):
        return pd.read_pickle(savepath)

    for var in ['posX', 'posY', 'posZ']:
        train_seq = data[data['type'] == 'train'][var].to_numpy().reshape((-1, 1))
        scaler = StandardScaler().fit(train_seq)
        seq = data[var].to_numpy().reshape((-1, 1))
        normalized = scaler.transform(seq)
        data.loc[:, var] = normalized[:, 0]
    print(data)


if __name__ == '__main__':
    data = load_thor_data()
    data = preprocess_thor_data(data)
