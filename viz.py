from data import input_manager
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def viz(data, actions):
    idx = pd.IndexSlice
    x = []
    for cluster in [0, 1]:
        actions_ = actions[actions['label'] == cluster].sample(n=5)
        for i, row in actions_.iterrows():
            rows = data.loc[row['session']]
            rows = rows[rows.index.isin(range(row['start_step'], row['end_step'] + 1))]
            x_ = rows.loc[:, idx[['depth'], ['RightHand']]]
            x_.reset_index(drop=True, inplace=True)
            x_['trial'] = i
            x_['cluster'] = cluster
            x.append(x_)
    x = pd.concat(x)
    x.columns = x.columns.get_level_values(0)
    x.reset_index(inplace=True)
    x = x.pivot(index='index', columns=['cluster', 'trial'], values='depth')
    sns.lineplot(data=x[0])
    plt.show()


if __name__ == '__main__':
    data = input_manager.load_cached('energy')
    data = input_manager.compute_relative(data)
    actions = input_manager.load_cached('actions_with_cluster_labels_64')
    viz(data, actions)
