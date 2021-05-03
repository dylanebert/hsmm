from nbc_bridge import load_nbc_data
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_motion(data):
    pos = data[['posX', 'posY', 'posZ']]
    vel = pos.diff().fillna(0)
    pos['motion'] = vel.apply(lambda row: np.sqrt(row['posX'] * row['posX'] + row['posY'] * row['posY'] + row['posZ'] * row['posZ']), axis=1)
    return pos


def smooth_motion(data):
    data['motion_smoothed'] = data['motion'].rolling(25).mean()
    return data


if __name__ == '__main__':
    nbc_data = load_nbc_data().iloc[:1000]
    data = get_motion(nbc_data)
    data = smooth_motion(data)
    sns.lineplot(x=data.index, y='motion', data=data)
    sns.lineplot(x=data.index, y='motion_smoothed', data=data)
    plt.show()
