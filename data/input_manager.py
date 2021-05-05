from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd


def load_cached(fname):
    import os
    NBC_ROOT = os.environ['NBC_ROOT']
    fpath = f'{NBC_ROOT}/cache/{fname}.p'
    data = pd.read_pickle(fpath)
    return data


def cache(data, fname):
    import os
    NBC_ROOT = os.environ['NBC_ROOT']
    fpath = f'{NBC_ROOT}/cache/{fname}.p'
    data.to_pickle(fpath)


def subsample(data, skip=9):
    indices = list(range(0, len(data), skip))
    data = data.iloc[indices]
    return data


def compute_relative(data):
    data = data.sort_index()
    idx = pd.IndexSlice
    head_pos = data.loc[:, idx[['posX', 'posY', 'posZ'], 'Head']].to_numpy()
    head_rot = data.loc[:, idx[['rotX', 'rotY', 'rotZ', 'rotW'], 'Head']].to_numpy()
    head_rot = R.from_quat(head_rot)
    head_yaw = head_rot.as_euler('xyz', degrees=True)
    head_yaw[:, [0, 2]] = 0
    head_yaw = R.from_euler('xyz', head_yaw, degrees=True)

    objs = data['posX'].columns.tolist()
    for obj in objs:
        pos = data.loc[:, idx[['posX', 'posY', 'posZ'], obj]].to_numpy()
        rel = head_yaw.apply(pos - head_pos, inverse=True)
        data['horizon', obj] = rel[:, 2]
        data['height', obj] = rel[:, 1]
        data['depth', obj] = rel[:, 0]

    return data


def compute_motion(data, params=['horizon', 'height', 'depth']):
    data = data.sort_index()
    sessions = data.index.unique(level='session').values
    objs = data['posX'].columns.tolist()
    idx = pd.IndexSlice
    for session in sessions:
        for obj in objs:
            pos = data.loc[session, idx[params, obj]]
            motion = pos.diff().fillna(0).to_numpy()
            for i, param in enumerate(params):
                data.loc[session, (f'motion_{param}', obj)] = motion[:, i]
            data.loc[session, ('motion', obj)] = np.linalg.norm(motion, axis=-1)

    return data


if __name__ == '__main__':
    data = load_cached('nbc_sub3')
