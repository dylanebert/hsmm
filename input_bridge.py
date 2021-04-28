import numpy as np
import pandas as pd
import os
import sys
import json


assert 'THOR_ROOT' in os.environ, 'set THOR_ROOT'
THOR_ROOT = os.environ['THOR_ROOT']


if __name__ == '__main__':
    dir = THOR_ROOT + '/data/'
    fnames = os.listdir(dir)
    for fname in fnames:
        fpath = os.path.join(dir, fname)
        df = pd.read_json(fpath, orient='index')
        print(df)
        break
