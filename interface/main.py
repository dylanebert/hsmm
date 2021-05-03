import os
import json
from flask import Flask, request, render_template
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
