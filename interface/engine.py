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
    controller.initialize(fpath)
    return 'success'

@app.route('/get_eval')
def get_eval():
    hsmm_wrapper = controller.hsmm_wrapper
    eval = OddManOut(hsmm_wrapper)
    questions = eval.questions
    answers = eval.answers
    for qidx in range(len(answers)):
        if answers[qidx] == None:
            print((qidx, len(answers)))
            return json.dumps((qidx, questions[qidx]))
    return 'done'

@app.route('/write_answer')
def write_answer():
    qidx = int(request.args.get('qidx'))
    answer = int(request.args.get('answer'))
    hsmm_wrapper = controller.hsmm_wrapper
    eval = OddManOut(hsmm_wrapper)
    eval.answers[qidx] = answer
    eval.save()
    return 'success'

@app.route('/get_sessions')
def get_sessions():
    sessions = list(controller.input_module.steps['dev'].keys())
    return json.dumps(sessions)

@app.route('/get_input_data')
def get_input_data():
    session = request.args.get('session')
    z, steps = controller.get_input_data(session)
    n_points, n_dim = z.shape
    pal = sns.color_palette('hls', n_dim).as_hex()
    datasets = []
    for i in range(n_dim):
        data = []
        for j in range(n_points):
            point = float(z[j,i])
            data.append(point)
        dataset = {
            'data': data,
            'label': str(i),
            'borderColor': pal[i],
            'fill': False
        }
        datasets.append(dataset)
    data = {'labels': steps.astype(str).tolist(), 'datasets': datasets}
    return json.dumps(data)

@app.route('/get_hsmm_input_encodings')
def get_hsmm_input_encodings():
    session = request.args.get('session')
    hsmm_wrapper = controller.hsmm_wrapper
    data = controller.get_hsmm_input_encodings(session, 'dev')
    pal = sns.color_palette('hls', data['label'].max() + 1).as_hex()
    data['x'] = data.apply(lambda row: row['encoding'][0], axis=1)
    try:
        data['y'] = data.apply(lambda row: row['encoding'][1], axis=1)
    except:
        data['y'] = 0
    datasets = []
    for label, rows in data.groupby('label'):
        dataset = {
            'label': label,
            'data': rows[['x', 'y', 'timestamp', 'start_step', 'end_step']].to_dict(orient='records'),
            'backgroundColor': pal[label]
        }
        datasets.append(dataset)
    return json.dumps(datasets)

@app.route('/get_predictions')
def get_predictions():
    session = request.args.get('session')
    predictions, steps = controller.get_predictions(session, 'dev')
    if steps.ndim == 1:
        steps = np.stack((steps, steps + 9), axis=-1)
    datasets = []
    pal = sns.color_palette('hls', controller.hsmm_wrapper.args['n_classes']).as_hex()
    for i, pred in enumerate(predictions):
        datasets.append({
            'data': [1],
            'label': int(pred),
            'backgroundColor': pal[pred],
            'start_step': int(steps[i][0]),
            'end_step': int(steps[i][-1]),
            'timestamp': controller.get_timestamp(session, steps[i][0])
        })
    return json.dumps(datasets)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
