from flask import Flask, request
import sys
sys.path.append('C:/Users/dylan/Documents')
sys.path.append('C:/Users/dylan/Documents/seg/hsmm')
from nbc.nbc import NBC
import controller
import json
import numpy as np
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def hello():
    return 'python engine running'

@app.route('/load_config')
def load_config():
    fpath = request.args.get('fpath')
    global args
    args = controller.deserialize(fpath)
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
    global nbc
    nbc = NBC(args)
    z, y = controller.get_encodings(args, type='dev')
    datasets = []
    labels = np.unique(y)
    pal = sns.color_palette('hls', len(labels)).as_hex()
    for i, label in enumerate(labels):
        data = []
        indices = np.arange(0, z.shape[0])[y==label]
        steps = list(nbc.steps['dev'].items())
        for j, elem in enumerate(z[y==label]):
            idx = int(indices[j])
            session = steps[idx][0][0]
            start_step, end_step = steps[idx][1][0], steps[idx][1][-1]
            data.append({'x': float(elem[0]), 'y': float(elem[1]), 'idx': int(indices[j]), 'session': session, 'start_step': str(start_step), 'end_step': str(end_step)})
        dataset = {'label': nbc.label_mapping[str(label)], 'data': data, 'backgroundColor': pal[i]}
        datasets.append(dataset)
    return json.dumps(datasets)

@app.route('/get_elem_by_idx')
def get_elem_by_idx():
    idx = int(request.args.get('idx'))
    global nbc
    assert nbc is not None
    keys = list(nbc.steps['dev'].keys())
    steps = list(nbc.steps['dev'].values())
    session = keys[idx][0]
    start_step, end_step = steps[idx][0], steps[idx][-1]
    return json.dumps([session, start_step, end_step])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
