from data import input_manager
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import pandas as pd


def get_cluster_labels(actions, x):
    km = TimeSeriesKMeans(n_clusters=8, metric='dtw').fit(x['train'])
    actions_split = {}
    for type in ['train', 'dev', 'test']:
        actions_split[type] = actions[actions['type'] == type]
        labels = km.predict(x[type])
        actions_split[type].loc[:, 'label'] = labels
    actions = pd.concat([actions_split[type] for type in ['train', 'dev', 'test']])
    return actions


def get_trajectory_vectors(data, actions):
    x = {}
    data_split = {}
    actions_split = {}
    idx = pd.IndexSlice
    for type in ['train', 'dev', 'test']:
        sessions = data[data['type'] == type].index.unique(level='session').tolist()
        data_split[type] = data[data['type'] == type]
        actions_split[type] = actions[actions['session'].isin(sessions)]
        actions_split[type].loc[:, 'type'] = type
        x[type] = []
        for _, row in actions_split[type].iterrows():
            rows = data.loc[row['session']]
            rows = rows[rows.index.isin(range(row['start_step'], row['end_step'] + 1))]
            x_ = rows.loc[:, idx[['height', 'depth'], ['LeftHand', 'RightHand']]]
            x[type].append(x_)
        x[type] = to_time_series_dataset(x[type])
        print(x[type].shape)
    actions = pd.concat([actions_split[type] for type in ['train', 'dev', 'test']])
    return x, actions


if __name__ == '__main__':
    data = input_manager.load_cached('energy')
    data = input_manager.compute_relative(data)
    actions = input_manager.load_cached('actions')
    x, actions = get_trajectory_vectors(data, actions)
    actions = get_cluster_labels(actions, x)
    input_manager.cache(actions, 'actions_with_cluster_labels')
