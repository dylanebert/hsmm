import json
from flask import Flask, request, render_template
import seaborn as sns
from data import input_manager
import pandas as pd


app = Flask(__name__)

df = input_manager.load_cached('nbc_sub3_energy')
actions = input_manager.load_cached('nbc_sub3_actions')


@app.route('/')
def index():
    return render_template('dataviewer.html')


@app.route('/get_sessions')
def get_sessions():
    sessions = df.index.unique(level='session').tolist()
    return json.dumps(sessions)


@app.route('/get_input_data')
def get_input_data():
    session = request.args.get('session')
    rows = df.loc[session]
    actions_ = actions[actions['session'] == session]
    print(actions_)
    idx = pd.IndexSlice
    labels = ['energy']
    z = rows.loc[:, idx[labels]].to_numpy()
    steps = rows.index.to_numpy()
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
