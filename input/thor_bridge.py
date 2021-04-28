import pandas as pd
import os
# from scipy.spatial.transform import Rotation as R


assert 'THOR_ROOT' in os.environ, 'set THOR_ROOT'
THOR_ROOT = os.environ['THOR_ROOT']


def load_thor_data():
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
            pos['type'] = type
            pos['session'] = fname.replace('.json', '')
            # rot_euler = df[['rotX', 'rotY', 'rotZ']].to_numpy()
            # rot = R.from_euler('xyz', rot_euler, degrees=True).as_quat()
            data.append(pos)
    data = pd.concat(data)
    print(data)


if __name__ == '__main__':
    load_thor_data()
