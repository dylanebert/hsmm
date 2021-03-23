import input_modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reduce(z):
    n_dim = z['train'].shape[-1]
    scaler = PCA(n_components=2).fit(z['train'].reshape((-1, n_dim)))
    for type in ['train', 'dev', 'test']:
        z[type] = scaler.transform(z[type].reshape((-1, n_dim))).reshape((z[type].shape[0], z[type].shape[1], 2))
    return z

if __name__ == '__main__':
    sessions = input_modules.InputModule.build_from_config('autoencoder_Apple')
    n = int(sessions.lengths['dev'][0])
    x = sessions.child.child.inference_module.child.child.z['dev'][:n]
    labels = np.any(x > 0, axis=(1, 2)).astype(int)
    print(np.mean(x[labels == 0]))
    print(np.mean(x[labels == 1]))
    np.set_printoptions(threshold=1e9)
    z = reduce(sessions.z)['dev'][0][:n]
    print(np.mean(z[labels == 0]))
    print(np.mean(z[labels == 1]))
    steps = next(iter(sessions.steps['dev'].values()))
    plt.scatter(x=z[:,0], y=z[:,1], c=labels)
    plt.show()
