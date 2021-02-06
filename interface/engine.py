from flask import Flask, request
import json
import numpy as np
import seaborn as sns
from colorutils import Color
import os
import sys

assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(HSMM_ROOT)
sys.path.append(NBC_ROOT)
from nbc_wrapper import NBCWrapper
import config
import autoencoder_bridge

app = Flask(__name__)

args = None
nbc_wrapper = None

@app.route('/')
def hello():
    return 'python engine running'

@app.route('/load_config')
def load_config():
    fpath = request.args.get('fpath')
    global args
    args = config.deserialize(fpath)
    autoencoder_bridge.initialize(args)
    return 'success'

@app.route('/get_args')
def get_args():
    global args
    assert args is not None
    return json.dumps(vars(args))

@app.route('/get_encodings')
def get_encodings():
    global args
    assert args is not None
    global nbc_wrapper
    nbc_wrapper = NBCWrapper(args)
    z = autoencoder_bridge.get_encodings(args, type='dev')
    y = nbc_wrapper.y['dev']
    datasets = []
    labels = np.unique(y)
    pal = sns.color_palette('hls', len(labels)).as_hex()
    for i, label in enumerate(labels):
        data = []
        print(label)
        print(z)
        print(y)
        indices = np.arange(0, z.shape[0])[y==label]
        steps = list(nbc_wrapper.nbc.steps['dev'].items())
        for j, elem in enumerate(z[y==label]):
            idx = int(indices[j])
            session = steps[idx][0][0]
            start_step, end_step = steps[idx][1][0], steps[idx][1][-1]
            data.append({'x': float(elem[0]), 'y': float(elem[1]), 'idx': int(indices[j]), 'session': session, 'start_step': str(start_step), 'end_step': str(end_step)})
        dataset = {'label': nbc_wrapper.nbc.label_mapping[str(label)], 'data': data, 'backgroundColor': pal[i]}
        datasets.append(dataset)
    return json.dumps(datasets)

@app.route('/get_elem_by_idx')
def get_elem_by_idx():
    idx = int(request.args.get('idx'))
    global nbc_wrapper
    assert nbc_wrapper is not None
    keys = list(nbc_wrapper.nbc.steps['dev'].keys())
    steps = list(nbc_wrapper.nbc.steps['dev'].values())
    session = keys[idx][0]
    start_step, end_step = int(steps[idx][0]), int(steps[idx][-1])
    x, x_ = autoencoder_bridge.get_reconstruction(args, type='dev')
    x, x_ = x[idx], x_[idx]
    seq_end = (x[:,0] == -1e9).argmax()
    if seq_end == 0:
        seq_end = x_.shape[0]
    x, x_ = x[:seq_end,:], x_[:seq_end,:]
    datasets = []
    pal = sns.color_palette('hls', x.shape[1]).as_hex()
    for j in range(x.shape[1]):
        data = []
        data_ = []
        for i in range(x.shape[0]):
            data.append(float(x[i,j]))
            data_.append(float(x_[i,j]))
        color = Color(hex=pal[j])
        hsv = color.hsv
        hue = hsv[0]
        sat = max(hsv[1] - .3, 0.)
        val = min(hsv[2] + .3, 1.)
        original_color = Color(hsv=(hue, sat, val))
        dataset = {'label': nbc_wrapper.nbc.feature_mapping[str(j)], 'data': data, 'fill': False, 'borderColor': original_color.hex}
        dataset_ = {'label': nbc_wrapper.nbc.feature_mapping[str(j)] + '_reconstr', 'data': data_, 'fill': False, 'borderColor': color.hex}
        datasets.append(dataset)
        datasets.append(dataset_)
    labels = [int(v) for v in np.arange(0, x.shape[0])]
    data = {'datasets': datasets, 'labels': labels}
    res = {'session': session, 'start_step': start_step, 'end_step': end_step, 'data': data}
    return json.dumps(res)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
