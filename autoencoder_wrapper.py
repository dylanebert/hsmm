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

    def __init__(self, args):
        self.args = args
        if isinstance(args.input_config, list):
            self.x = {'train': [], 'dev': [], 'test': []}
            self.x_trim = {'train': [], 'dev': [], 'test': []}
            for cfg in args.input_config:
                nbc_args = config.deserialize(cfg)
                nbc_wrapper = NBCWrapper(nbc_args)
                x = nbc_wrapper.x
                x_trim = nbc_wrapper.x_trim
                for type in ['train', 'dev', 'test']:
                    self.x[type].append(x[type])
                    self.x_trim[type].append(x_trim[type])
            for type in ['train', 'dev', 'test']:
                self.x[type] = np.concatenate(self.x[type], axis=0)
                self.x_trim[type] = np.concatenate(self.x_trim[type], axis=0)
        else:
            assert isinstance(args.input_config, str)
            nbc_args = config.deserialize(args.input_config)
            self.nbc_wrapper = NBCWrapper(nbc_args)
            self.x = self.nbc_wrapper.x
            self.x_trim = self.nbc_wrapper.x_trim
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
            self.vae(self.x['train'])
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
        _, seq_len, input_dim = self.x_trim['train'].shape
        train_dset = tf.data.Dataset.from_tensor_slices(self.x_trim['train']).batch(self.args.vae_batch_size)
        dev_dset = tf.data.Dataset.from_tensor_slices(self.x_trim['dev']).batch(self.args.vae_batch_size)

        self.vae = VAE(seq_len, input_dim, self.args.vae_hidden_size, self.args.vae_beta, self.args.vae_warm_up_iters)
        self.vae.compile(optimizer='adam')
        tmp_path = NBC_ROOT + 'cache/autoencoder/tmp_{}.h5'.format(str(uuid.uuid1()))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=25, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(tmp_path, save_best_only=True, verbose=1)
        ]
        self.vae.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)
        self.vae(self.x_trim['train'][:10])
        self.vae.load_weights(tmp_path)

    def get_encodings(self):
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
        self.get_encodings()
        self.cache()

    def reduced_encodings(self):
        reducer = PCA(n_components=2).fit(self.encodings['train'])
        encodings = {}
        for type in ['train', 'dev', 'test']:
            encodings[type] = reducer.transform(self.encodings[type])
        return encodings

class AutoencoderUnifiedCombiner(AutoencoderWrapper):
    def __init__(self, args):
        self.args = args
        nbc_args = config.deserialize(args.input_config[0])
        self.nbc_wrapper = NBCWrapper(nbc_args)
        self.load()

    def load(self):
        if self.try_load_cached():
            return
        self.get_encodings()
        self.cache()

    def try_load_cached(self):
        savefile = config.find_savefile(self.args, 'autoencoder')
        if savefile is None:
            return False
        input_path = NBC_ROOT + 'cache/autoencoder/{}_unified_inputs.json'.format(savefile)
        encodings_path = NBC_ROOT + 'cache/autoencoder/{}_unified_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'cache/autoencoder/{}_unified_reconstructions.json'.format(savefile)
        if not os.path.exists(input_path) or not os.path.exists(encodings_path) or not os.path.exists(reconstructions_path):
            return False
        with open(input_path) as f:
            self.x = json.load(f)
        with open(encodings_path) as f:
            self.encodings = json.load(f)
        with open(reconstructions_path) as f:
            self.reconstructions = json.load(f)
        for type in ['train', 'dev', 'test']:
            self.x[type] = np.array(self.x[type])
            self.encodings[type] = np.array(self.encodings[type])
            self.reconstructions[type] = np.array(self.reconstructions[type])
        print('loaded cached autoencoder unified wrapper')
        return True

    def cache(self):
        savefile = config.generate_savefile(self.args, 'autoencoder')
        input_path = NBC_ROOT + 'cache/autoencoder/{}_unified_inputs.json'.format(savefile)
        encodings_path = NBC_ROOT + 'cache/autoencoder/{}_unified_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'cache/autoencoder/{}_unified_reconstructions.json'.format(savefile)
        with open(input_path, 'w+') as f:
            serialized = {}
            for type in ['train', 'dev', 'test']:
                serialized[type] = self.x[type].tolist()
            json.dump(serialized, f)
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
        print('cached autoencoder unified wrapper')

    def get_encodings(self):
        self.autoencoder_wrapper = AutoencoderWrapper(args)
        assert self.autoencoder_wrapper.try_load_cached(load_model=True)
        self.x = {'train': [], 'dev': [], 'test': []}
        for cfg in args.input_config:
            print(cfg)
            nbc_args = config.deserialize(cfg)
            nbc_wrapper = NBCWrapper(nbc_args)
            for type in ['train', 'dev', 'test']:
                self.x[type].append(nbc_wrapper.x[type])

        self.encodings = {}
        self.reconstructions = {}
        for type in ['train', 'dev', 'test']:
            x = np.stack(self.x[type], axis=1)
            magnitudes = np.linalg.norm(x, axis=(2, 3))
            max_indices = np.argmax(magnitudes, axis=1)
            print(np.unique(max_indices, return_counts=True)[1])
            max_x = []
            for i in range(x.shape[0]):
                max_x.append(x[i, max_indices[i], :])
            max_x = np.array(max_x)
            z, _ = self.autoencoder_wrapper.vae.encode(max_x)
            x_reconstr = self.autoencoder_wrapper.vae(max_x)

            self.x[type] = max_x
            self.encodings[type] = z
            self.reconstructions[type] = x_reconstr

class AutoencoderMaxWrapper(AutoencoderWrapper):
    def __init__(self, configs, add_indices=False):
        args = config.deserialize(configs[0])
        self.nbc_wrapper = AutoencoderWrapper(args).nbc_wrapper

        print('getting max autoencoder encodings')
        self.x = {'train': [], 'dev': [], 'test': []}
        self.encodings = {'train': [], 'dev': [], 'test': []}
        self.reconstructions = {'train': [], 'dev': [], 'test': []}
        for cfg in configs:
            args = config.deserialize(cfg)
            aw = AutoencoderWrapper(args)
            for type in ['train', 'dev', 'test']:
                self.x[type].append(aw.x[type])
                self.encodings[type].append(aw.encodings[type])
                self.reconstructions[type].append(aw.reconstructions[type])
        for type in ['train', 'dev', 'test']:
            x = np.stack(self.x[type], axis=1)
            encodings = np.stack(self.encodings[type], axis=1)
            reconstr = np.stack(self.reconstructions[type], axis=1)

            magnitudes = np.linalg.norm(x, axis=(2, 3))
            max_indices = np.argmax(magnitudes, axis=1)
            max_x = []
            max_encodings = []
            max_reconstr = []
            for i in range(encodings.shape[0]):
                max_x.append(x[i, max_indices[i], :])
                max_encodings.append(encodings[i, max_indices[i], :])
                max_reconstr.append(reconstr[i, max_indices[i], :])
            if add_indices:
                one_hot = []
                n_objs = len(configs)
                for i in range(encodings.shape[0]):
                    v = magnitudes[i, max_indices[i]]
                    vec = np.zeros((n_objs,))
                    if v > 0.:
                        vec[max_indices[i]] = 1
                    one_hot.append(vec)
                one_hot = np.array(one_hot)
                max_encodings = np.concatenate((max_encodings, one_hot), axis=-1)

            max_x = np.array(max_x)
            max_encodings = np.array(max_encodings)
            max_reconstr = np.array(max_reconstr)

            self.x[type] = max_x
            self.encodings[type] = max_encodings
            self.reconstructions[type] = max_reconstr


if __name__ == '__main__':
    args = config.deserialize('autoencoder_objs')
    wrapper = AutoencoderUnifiedCombiner(args)
