from nbc import NBC, obj_names
from data.config import NBCConfig
import pandas as pd


valid_objs = [name for name in obj_names if name not in ['Head', 'LeftHand', 'RightHand']]


def load_nbc_data():
    nbc_config = NBCConfig({'nbc_features': ['posX:RightHand', 'posY:RightHand', 'posZ:RightHand']})
    nbc = NBC(nbc_config)
    data = []
    for type in ['train', 'dev', 'test']:
        for key, seq in nbc.features[type].items():
            for i in range(seq.shape[0]):
                row = {
                    'session': key[0],
                    'posX': seq[i, 0],
                    'posY': seq[i, 1],
                    'posZ': seq[i, 2],
                    'step': nbc.steps[type][key][i],
                    'type': type
                }
                data.append(row)
    data = pd.DataFrame(data)
    return data


if __name__ == '__main__':
    data = load_nbc_data()
    print(data)
