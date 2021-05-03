import json
from flask import Flask, request, render_template
import seaborn as sns
from data import nbc_bridge
from data import input_manager
import numpy as np
import pandas as pd


app = Flask(__name__)
df = nbc_bridge.load_nbc_data()
df = input_manager.subsample(df)
df = input_manager.compute_depth(df)


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
    idx = pd.IndexSlice
    labels = ['height', 'depth']
    z = rows.loc[:, idx[labels, 'RightHand']].to_numpy()
    print(z.shape)
    steps = np.array([index[0] for index in rows.index.values])
    print(steps)
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
            'label': labels[i],
            'borderColor': pal[i],
            'fill': False
        }
        datasets.append(dataset)
    data = {'labels': steps.astype(str).tolist(), 'datasets': datasets}
    return json.dumps(data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
