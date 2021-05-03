import os
import json
from flask import Flask, request, render_template
import numpy as np
import seaborn as sns
from data import nbc_bridge


assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']


app = Flask(__name__)
df = nbc_bridge.load_nbc_data()


@app.route('/')
def index():
    return render_template('dataviewer.html')


@app.route('/get_sessions')
def get_sessions():
    sessions = list(df['session'].unique())
    return json.dumps(sessions)


@app.route('/get_input_data')
def get_input_data():
    session = request.args.get('session')
    rows = df[df['session'] == session]
    z = rows[['posX', 'posY', 'posZ']].to_numpy()
    steps = rows['step'].to_numpy()
    n_points, n_dim = z.shape
    pal = sns.color_palette('hls', n_dim).as_hex()
    datasets = []
    for i in range(n_dim):
        points = []
        for j in range(n_points):
            point = float(z[j, i])
            points.append(point)
        dataset = {
            'data': points,
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
    data = controller.get_hsmm_input_encodings(session, 'dev')
    pal = sns.color_palette('hls', data['label'].max() + 1).as_hex()
    data['x'] = data.apply(lambda row: row['encoding'][0], axis=1)
    try:
        data['y'] = data.apply(lambda row: row['encoding'][1], axis=1)
    except Exception:
        data['y'] = 0
    datasets = []
    for label, rows in data.groupby('label'):
        data = rows[['x', 'y', 'timestamp', 'start_step', 'end_step']]
        dataset = {
            'label': label,
            'data': data.to_dict(orient='records'),
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
    n_classes = controller.HSMM_WRAPPER.args['n_classes']
    pal = sns.color_palette('hls', n_classes).as_hex()
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
