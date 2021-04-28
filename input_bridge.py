import numpy as np
import os
import sys


assert 'THOR_ROOT' in os.environ, 'set THOR_ROOT'
THOR_ROOT = os.environ['THOR_ROOT']


if __name__ == '__main__':
    dir = THOR_ROOT + '/data/'
    fnames = os.listdir(dir)
    for fname in fnames:
        fpath = os.path.join(dir, fname)
        print(fpath)