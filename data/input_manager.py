from nbc_bridge import load_nbc_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.spatial.rotation as R
from sklearn.preprocessing import MinMaxScaler


def get_depth(data):
    head_rot = data[['']]


def get_motion(data):
    pos = data[['posX', 'posY', 'posZ']]
    vel = pos.diff().fillna(0).to_numpy()
    vel = np.linalg.norm(vel, axis=-1)
    data['motion'] = vel
    return data


if __name__ == '__main__':
    nbc_data = load_nbc_data().iloc[:1000]
    data = get_depth(data)
