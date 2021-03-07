from flask import Flask, request
import json
import numpy as np
import pandas as pd
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
import controller
from eval import OddManOut

app = Flask(__name__)
args = None

@app.route('/')
def hello():
    return 'python engine running'

@app.route('/load_config')
def load_config():
    fpath = request.args.get('fpath')
    global args
    args = config.deserialize(fpath)
    if args.nbc_train_sequencing == 'session':
        controller.initialize(args, 'hsmm_direct')
    else:
        controller.initialize(args, 'hsmm')
    return 'success'

@app.route('/get_eval')
def get_eval():
    global args
    assert args is not None
    nbc_wrapper = controller.nbc_wrapper
    hsmm_wrapper = controller.hsmm_wrapper
    eval = OddManOut(nbc_wrapper, hsmm_wrapper)
    questions = eval.questions
    answers = eval.answers
    for qidx in range(len(answers)):
        if answers[qidx] == None:
            return json.dumps((qidx, questions[qidx]))
    return 'done'

@app.route('/write_answer')
def write_answer():
    global args
    assert args is not None
    qidx = int(request.args.get('qidx'))
    answer = int(request.args.get('answer'))
    nbc_wrapper = controller.nbc_wrapper
    hsmm_wrapper = controller.hsmm_wrapper
    eval = OddManOut(nbc_wrapper, hsmm_wrapper)
    eval.answers[qidx] = answer
    eval.update_answers()
    return 'success'

@app.route('/get_sessions')
def get_sessions():
    sessions = controller.get_sessions('dev')
    return json.dumps(sessions)

@app.route('/get_encodings')
def get_encodings():
    if controller.nbc_wrapper is None:
        return 'na'
    global args
    assert args is not None
    session = request.args.get('session')
    nbc_wrapper = controller.nbc_wrapper
    data = controller.get_encodings(args, session=session, type='dev')
    pal = sns.color_palette('hls', data['label'].max() + 1).as_hex()
    data['x'] = data.apply(lambda row: row['encoding'][0], axis=1)
    data['y'] = data.apply(lambda row: row['encoding'][1], axis=1)
    datasets = []
    for label, rows in data.groupby('label'):
        dataset = {
            'label': label,
            'data': rows[['x', 'y', 'idx', 'start_step', 'end_step']].to_dict(orient='records'),
            'backgroundColor': pal[label]
        }
        datasets.append(dataset)
    return json.dumps(datasets)

@app.route('/get_encoding_by_idx')
def get_encoding_by_idx():
    #get encoding by index
    nbc = controller.nbc
    nbc_wrapper = controller.nbc_wrapper
    if nbc is not None:
        idx = int(request.args.get('idx'))
        keys = list(nbc.steps['dev'].keys())
        steps = list(nbc.steps['dev'].values())
        start_step, end_step = int(steps[idx][0]), int(steps[idx][-1])
        sessions = np.array([key[0] for key in keys])
        session_start_step = start_step
        timestamp = (start_step - session_start_step) / 90.
        res = {'start_step': start_step, 'end_step': end_step, 'timestamp': timestamp}
        return json.dumps(res)
    else:
        assert nbc_wrapper is not None
        idx = int(request.args.get('idx'))
        x, x_ = controller.get_reconstruction(args, idx, type='dev')
        keys = list(nbc_wrapper.nbc.steps['dev'].keys())
        steps = list(nbc_wrapper.nbc.steps['dev'].values())
        start_step, end_step = int(steps[idx][0]), int(steps[idx][-1])
        sessions = np.array([key[0] for key in keys])
        session_start_step = keys[(sessions == keys[idx][0]).argmax()][1]
        timestamp = (start_step - session_start_step) / 90.

        #convert to json-friendly representation
        datasets = []
        pal = sns.color_palette('hls', x.shape[1]).as_hex()
        for j in range(x.shape[1]):
            data = []
            data_ = []
            for i in range(x.shape[0]):
                data.append(float(x[i, j]))
                data_.append(float(x_[i, j]))
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
        res = {'start_step': start_step, 'end_step': end_step, 'data': data, 'timestamp': timestamp}
        return json.dumps(res)

@app.route('/get_hsmm_predictions')
def get_hsmm_predictions():
    global args
    assert args is not None
    session = request.args.get('session')
    predictions, indices = controller.get_predictions(args, session, 'dev')
    if not isinstance(indices, list):
        indices = [indices] * len(predictions)
    datasets = []
    pal = sns.color_palette('hls', args.sm_n_classes).as_hex()
    for i, pred in enumerate(predictions):
        datasets.append({
            'data': [1],
            'label': int(pred),
            'backgroundColor': pal[pred],
            'idx': int(indices[i])
        })
    return json.dumps(datasets)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
