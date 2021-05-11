from data import input_manager
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import scipy.signal as signal
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_distance_to_clusters(x, centers):
    centers = np.concatenate(centers, axis=0)
    dists = cdist(x, centers, metric='sqeuclidean')
    min_dist = np.amin(dists)
    min_idx = np.argmin(dists)
    return min_dist, min_idx


class DynamicClustering:
    def __init__(self, delta):
        self.delta = delta

    def fit(self, x):
        self.centers = []
        self.radius = []
        self.idx = []
        self.M = []
        self.NSamples = []

        n_frames, n_dims = x.shape
        X = np.split(x, n_frames, axis=0)
        idx_saver = 0
        ii = 0
        dr = n_dims * self.delta

        for x in tqdm(X):
            ii += 1
            if not self.centers:
                self.centers.append(x)
                self.idx.append(idx_saver)
                self.radius.append(0.)
                self.NSamples.append(1.)
                self.M.append(x*x)
            else:
                min_dist, min_idx = compute_distance_to_clusters(x, self.centers)
                if min_dist > max(self.radius[min_idx], dr):
                    self.centers.append(x)
                    self.NSamples.append(1.)
                    self.M.append(x*x)
                    self.radius.append(0.)
                    idx_saver += 1
                    self.idx.append(idx_saver)
                else:
                    self.NSamples[min_idx] += 1
                    self.centers[min_idx] = self.centers[min_idx] + (x - self.centers[min_idx]) / self.NSamples[min_idx]
                    self.M[min_idx] = self.M[min_idx] + (x*x - self.M[min_idx]) / self.NSamples[min_idx]
                    Sigma = self.M[min_idx] - self.centers[min_idx] ** 2
                    self.radius[min_idx] = np.sum(Sigma)
                    self.idx.append(min_idx)
        return self

    def transform(self, x):
        X = np.split(x, x.shape[0], axis=0)
        idx = []
        for x in tqdm(X):
            min_dist, min_idx = compute_distance_to_clusters(x, self.centers)
            idx.append(min_idx)
        return idx


def get_motion_energy(x):
    transitions = np.sum(np.abs(np.diff(x)) > 1e-6)
    return transitions / float(x.shape[0])


def compute_motion_energy(data, W=30, sigma=5, delta=3e-2):
    idx = pd.IndexSlice
    data_split = {}
    z = {}
    for type in ['train', 'dev', 'test']:
        data_split[type] = data[data['type'] == type]
        z[type] = data_split[type].loc[:, idx[['posX', 'posY', 'posZ'], ['LeftHand', 'RightHand']]].to_numpy()
    clustering = DynamicClustering(delta).fit(z['train'])
    clusters = {}
    for type in ['train', 'dev', 'test']:
        clusters[type] = clustering.transform(z[type])
        data_split[type].loc[:, 'cluster'] = clusters[type]
    data = pd.concat([data_split[type] for type in ['train', 'dev', 'test']]).sort_index()
    x = data.loc[:, 'cluster'].to_numpy()
    x_pad = np.pad(x, W // 2, mode='edge')
    me_curve = np.array([get_motion_energy(x_pad[i:i+W]) for i in range(x.shape[0])])
    me_curve = gaussian_filter1d(me_curve, sigma)
    data.loc[:, 'energy'] = me_curve
    return data


def compute_action_boundaries(data, peak_distance=30, max_peak_width=100):
    me_curve = data.loc[:, 'energy'].to_numpy()
    peak_idx, _ = signal.find_peaks(me_curve, distance=peak_distance)
    peak_width = signal.peak_widths(me_curve, peak_idx, rel_height=1, wlen=max_peak_width)[0].astype(int)
    n_peaks = peak_idx.shape[0]
    actions = []
    index = data.index.tolist()
    for i in range(n_peaks):
        start_idx = max(0, peak_idx[i] - peak_width[i] // 2)
        end_idx = min(peak_idx[i] + peak_width[i] // 2, me_curve.shape[0] - 1)
        start = index[start_idx]
        end = index[end_idx]
        if start[0] == end[0]:
            actions.append({
                'session': start[0],
                'start_step': start[1],
                'end_step': end[1]
            })
    actions = pd.DataFrame(actions)
    return data, actions


if __name__ == '__main__':
    data = input_manager.load_cached('nbc_sub3')
    data = compute_motion_energy(data)
    data, actions = compute_action_boundaries(data)
    input_manager.cache(data, 'nbc_sub3_energy')
    input_manager.cache(actions, 'nbc_sub3_actions')
    print(actions)
