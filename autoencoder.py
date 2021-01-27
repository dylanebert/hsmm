import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import argparse
import random
import scipy
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import json
import os

HSMM_ROOT = 'C:/Users/dylan/Documents/seg/hsmm/'

class VAE(tf.keras.models.Model):
    def __init__(self, seq_len, input_dim, hidden_dim, beta=1):
        super(VAE, self).__init__()
        prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(hidden_dim), scale=1), reinterpreted_batch_ndims=1)
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(seq_len, input_dim)),
            tf.keras.layers.LSTM(hidden_dim),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(hidden_dim)),
            tfp.layers.MultivariateNormalTriL(hidden_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=beta))
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(hidden_dim,)),
            tf.keras.layers.RepeatVector(seq_len),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim, activation='sigmoid'))
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AutoencoderWrapper:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--vae_hidden_size', type=int, default=8)
        parser.add_argument('--vae_batch_size', type=int, default=10)
        parser.add_argument('--vae_beta', type=int, default=10)

    def __init__(self, args, nbc_wrapper):
        self.args = args
        self.nbc_wrapper = nbc_wrapper
        self.x, self.y = self.nbc_wrapper.x, self.nbc_wrapper.y
        self.make_paths()
        self.get_autoencoder()

    def make_paths(self):
        for dir in ['weights', 'encodings', 'reconstructions']:
            dirpath = '{}autoencoder/{}'.format(HSMM_ROOT, dir)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    def args_to_id(self):
        args_dict = {'nbc_id': self.nbc_wrapper.nbc.args_to_id()}
        for k in ['vae_hidden_size', 'vae_batch_size', 'vae_beta', 'nbc_output_type', 'nbc_preprocessing']:
            assert k in vars(self.args), k
            args_dict[k] = vars(self.args)[k]
        return json.dumps(args_dict)

    def try_load_model(self):
        args_id = self.args_to_id()
        keypath = HSMM_ROOT + 'autoencoder/keys.json'
        if not os.path.exists(keypath):
            return False
        with open(keypath) as f:
            keys = json.load(f)
        if args_id not in keys:
            return False
        fid = keys[args_id]
        weights_path = HSMM_ROOT + 'autoencoder/weights/{}.h5'.format(fid)
        encodings_path = HSMM_ROOT + 'autoencoder/encodings/{}.json'.format(fid)
        reconstructions_path = HSMM_ROOT + 'autoencoder/reconstructions/{}.json'.format(fid)
        self.vae(self.x['train'])
        self.vae.load_weights(weights_path)
        with open(encodings_path) as f:
            self.encodings = json.load(f)
        with open(reconstructions_path) as f:
            self.reconstructions = json.load(f)
        for type in ['train', 'dev', 'test']:
            self.encodings[type] = np.array(self.encodings[type])
            self.reconstructions[type] = np.array(self.reconstructions[type])
        return True

    def save_model(self):
        args_id = self.args_to_id()
        keypath = HSMM_ROOT + 'autoencoder/keys.json'
        if os.path.exists(keypath):
            with open(keypath) as f:
                keys = json.load(f)
        else:
            keys = {}
        fid = str(uuid.uuid1())
        weights_path = HSMM_ROOT + 'autoencoder/weights/{}.h5'.format(fid)
        encodings_path = HSMM_ROOT + 'autoencoder/encodings/{}.json'.format(fid)
        reconstructions_path = HSMM_ROOT + 'autoencoder/reconstructions/{}.json'.format(fid)
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
        keys[args_id] = fid
        with open(keypath, 'w+') as f:
            json.dump(keys, f)
        print('saved autoencoder')

    def train_autoencoder(self):
        _, seq_len, input_dim = self.x['train'].shape
        train_dset = tf.data.Dataset.from_tensor_slices(self.x['train']).batch(self.args.vae_batch_size)
        dev_dset = tf.data.Dataset.from_tensor_slices(self.x['dev']).batch(self.args.vae_batch_size)

        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        self.vae.compile(optimizer='adam', loss=negative_log_likelihood)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(HSMM_ROOT + 'autoencoder/weights/tmp.h5', save_best_only=True, verbose=1)
        ]
        self.vae.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)
        self.vae(self.x['train'])
        self.vae.load_weights(HSMM_ROOT + 'autoencoder/weights/tmp.h5')
        self.encodings = {}
        self.reconstructions = {}
        for type in ['train', 'dev', 'test']:
            z = self.vae.encoder(self.x[type])
            z = z.mean()
            x_ = self.vae(self.x[type])
            self.encodings[type] = z.numpy()
            self.reconstructions[type] = x_.numpy()

    def get_autoencoder(self):
        _, seq_len, input_dim = self.x['train'].shape
        self.vae = VAE(seq_len, input_dim, self.args.vae_hidden_size, beta=self.args.vae_beta)
        if self.try_load_model():
            print('loaded saved model')
            return
        self.train_autoencoder()
        self.save_model()
