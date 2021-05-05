from data import input_manager
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.signal as signal
import numpy as np
import pandas as pd


def energy_motion_pooling(data, n_clusters=32):
    idx = pd.IndexSlice
    data = data.iloc[:10000]
    z = data.loc[:, idx[['height', 'depth'], ['LeftHand', 'RightHand']]].to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(z)
    x = kmeans.labels_
    
    return data


if __name__ == '__main__':
    data = input_manager.load_cached('nbc_sub3_rel')
    data = energy_motion_pooling(data)
