import pandas as pd


def read_frames(fpath):
    with open(fpath) as f:
        lines = '[' + ','.join(f.readlines()) + ']'
    df = pd.read_json(lines)
    df.set_index('frame', drop=True, inplace=True)
    return df


def subsample(data, skip=9):
    indices = list(range(0, len(data), skip))
    data = data.iloc[indices]
    return data


if __name__ == '__main__':
    dir = r'C:\Users\dylan\AppData\LocalLow\DefaultCompany\Simulated\data'
    timestring = '637599834313986900-a312cbb0-1ae5-4267-810a-859d1e269767'
    fpath = f'{dir}/{timestring}/frames.json'
    df = read_frames(fpath)
    df = subsample(df)
