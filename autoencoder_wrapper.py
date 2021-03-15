import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from autoencoder import VAE
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import uuid

assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
import config
from nbc_wrapper import NBCWrapper

class AutoencoderWrapper:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--vae_hidden_size', type=int, default=8)
        parser.add_argument('--vae_batch_size', type=int, default=10)
        parser.add_argument('--vae_beta', type=int, default=10)
        parser.add_argument('--vae_warm_up_iters', type=int, default=1000)

    def __init__(self, args, nbc_wrapper):
        self.args = args
        self.nbc_wrapper = nbc_wrapper
        self.x = nbc_wrapper.x
        self.get_autoencoder()

    def try_load_cached(self, load_model=False):
        savefile = config.find_savefile(self.args, 'autoencoder')
        if savefile is None:
            return False
        weights_path = NBC_ROOT + 'cache/autoencoder/{}_weights.h5'.format(savefile)
        encodings_path = NBC_ROOT + 'cache/autoencoder/{}_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'cache/autoencoder/{}_reconstructions.json'.format(savefile)
        if not os.path.exists(weights_path) or not os.path.exists(encodings_path) or not os.path.exists(reconstructions_path):
            return False
        if load_model:
            _, seq_len, input_dim = self.x['train'].shape
            self.vae = VAE(seq_len, input_dim, self.args.vae_hidden_size, self.args.vae_beta, self.args.vae_warm_up_iters)
            self.vae(self.nbc_wrapper.x['train'])
            self.vae.load_weights(weights_path)
        with open(encodings_path) as f:
            self.encodings = json.load(f)
        with open(reconstructions_path) as f:
            self.reconstructions = json.load(f)
        for type in ['train', 'dev', 'test']:
            self.encodings[type] = np.array(self.encodings[type])
            self.reconstructions[type] = np.array(self.reconstructions[type])
        print('loaded cached autoencoder')
        return True

    def cache(self):
        savefile = config.generate_savefile(self.args, 'autoencoder')
        weights_path = NBC_ROOT + 'cache/autoencoder/{}_weights.h5'.format(savefile)
        encodings_path = NBC_ROOT + 'cache/autoencoder/{}_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'cache/autoencoder/{}_reconstructions.json'.format(savefile)
        self.vae.save_weights(weights_path)
        with open(encodings_path, 'w+') as f:
            serialized = {}
            for type in ['train', 'dev', 'test']:
                serialized[type] = self.encodings[type].tolist()
            json.dump(serialized, f)
        with open(reconstructions_path, 'w+') as f:
            serialized = {}
            for type in ['train', 'dev', 'test']:
                serialized[type] = self.reconstructions[type].tolist()
            json.dump(serialized, f)
        print('cached autoencoder')

    def train_autoencoder(self):
        nbc_wrapper_trim = NBCWrapper(self.args, True)
        x = nbc_wrapper_trim.x
        _, seq_len, input_dim = x['train'].shape
        train_dset = tf.data.Dataset.from_tensor_slices(x['train']).batch(self.args.vae_batch_size)
        dev_dset = tf.data.Dataset.from_tensor_slices(x['dev']).batch(self.args.vae_batch_size)

        self.vae = VAE(seq_len, input_dim, self.args.vae_hidden_size, self.args.vae_beta, self.args.vae_warm_up_iters)
        self.vae.compile(optimizer='adam')
        tmp_path = NBC_ROOT + 'cache/autoencoder/tmp_{}.h5'.format(str(uuid.uuid1()))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(tmp_path, save_best_only=True, verbose=1)
        ]
        self.vae.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)
        self.vae(x['train'])
        self.vae.load_weights(tmp_path)

        self.encodings = {}
        self.reconstructions = {}
        for type in ['train', 'dev', 'test']:
            z, _ = self.vae.encode(self.x[type])
            x_ = self.vae(self.x[type])
            self.encodings[type] = z.numpy()
            self.reconstructions[type] = x_.numpy()

    def get_autoencoder(self):
        if self.try_load_cached():
            return
        self.train_autoencoder()
        self.cache()

    def reduced_encodings(self):
        reducer = PCA(n_components=2).fit(self.encodings['train'])
        encodings = {}
        for type in ['train', 'dev', 'test']:
            encodings[type] = reducer.transform(self.encodings[type])
        return encodings

class AutoencoderConcatWrapper(AutoencoderWrapper):
    def __init__(self, configs):
        print('concatenating autoencoder encodings')
        self.x = {'train': [], 'dev': [], 'test': []}
        self.encodings = {'train': [], 'dev': [], 'test': []}
        self.reconstructions = {'train': [], 'dev': [], 'test': []}
        for cfg in configs:
            print(cfg)
            args = config.deserialize(cfg)
            nw = NBCWrapper(args)
            aw = AutoencoderWrapper(args, nw)
            for type in ['train', 'dev', 'test']:
                self.x[type].append(aw.x[type])
                self.encodings[type].append(aw.encodings[type])
                self.reconstructions[type].append(aw.reconstructions[type])
        for type in ['train', 'dev', 'test']:
            self.x[type] = np.concatenate(self.x[type], axis=-1)
            self.encodings[type] = np.concatenate(self.encodings[type], axis=-1)
            self.reconstructions[type] = np.concatenate(self.reconstructions[type], axis=-1)


if __name__ == '__main__':
    configs = ['hsmm_Apple_cov=0.1', 'hsmm_Ball_cov=0.1', 'hsmm_Banana_cov=1.0']
    wrapper = AutoencoderConcatWrapper(configs)
