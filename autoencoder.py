import numpy as np
import tensorflow as tf
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

assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
import config

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.math.exp(.5 * z_log_var) * epsilon

class VAE(tf.keras.models.Model):
    def __init__(self, seq_len, input_dim, hidden_dim, beta, warm_up_iters):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.beta = tf.cast(beta, tf.float32)
        self.warm_up_iters = tf.cast(warm_up_iters, tf.float32)
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(seq_len, input_dim)),
            tf.keras.layers.LSTM(hidden_dim),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(hidden_dim * 2)
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(hidden_dim,)),
            tf.keras.layers.RepeatVector(seq_len),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))
        ])
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_reconstr_loss = tf.keras.metrics.Mean(name='reconstr_loss')
        self.train_kl_loss = tf.keras.metrics.Mean(name='kl_loss')
        self.dev_loss = tf.keras.metrics.Mean(name='dev_loss')

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z)

    def encode(self, x):
        z_mean, z_log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(z_log_var * .5) + z_mean

    def compute_loss(self, x):
        def log_normal_pdf(sample, mean, logvar):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=1)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstr = self.decoder(z)
        logpx_z = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_reconstr), axis=1)
        logpz = log_normal_pdf(z, 0., 1.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        reconstr_loss = tf.reduce_mean(logpx_z)
        kl_loss = tf.reduce_mean(logqz_x - logpz)
        return reconstr_loss, kl_loss

    @tf.function
    def train_step(self, x):
        beta = self.beta * tf.math.minimum(tf.cast(self.optimizer.iterations, tf.float32) / self.warm_up_iters, tf.cast(1., tf.float32))
        with tf.GradientTape() as tape:
            reconstr_loss, kl_loss = self.compute_loss(x)
            loss = reconstr_loss + kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_reconstr_loss(reconstr_loss)
        self.train_kl_loss(kl_loss)
        return {
            'loss': self.train_loss.result(),
            'reconstr_loss': self.train_reconstr_loss.result(),
            'kl_loss': self.train_kl_loss.result(),
            'beta': beta
        }

    @tf.function
    def test_step(self, x):
        loss = self.compute_loss(x)
        self.dev_loss(loss)
        return {
            'loss': self.dev_loss.result(),
        }

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
        self.x, self.y = self.nbc_wrapper.x, self.nbc_wrapper.y
        self.get_autoencoder()

    def try_load_cached(self):
        savefile = config.find_savefile(self.args, 'autoencoder')
        if savefile is None:
            return False
        weights_path = NBC_ROOT + 'tmp/autoencoder/{}_weights.h5'.format(savefile)
        encodings_path = NBC_ROOT + 'tmp/autoencoder/{}_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'tmp/autoencoder/{}_reconstructions.json'.format(savefile)
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
        weights_path = NBC_ROOT + 'tmp/autoencoder/{}_weights.h5'.format(savefile)
        encodings_path = NBC_ROOT + 'tmp/autoencoder/{}_encodings.json'.format(savefile)
        reconstructions_path = NBC_ROOT + 'tmp/autoencoder/{}_reconstructions.json'.format(savefile)
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
        _, seq_len, input_dim = self.x['train'].shape
        train_dset = tf.data.Dataset.from_tensor_slices(self.x['train']).batch(self.args.vae_batch_size)
        dev_dset = tf.data.Dataset.from_tensor_slices(self.x['dev']).batch(self.args.vae_batch_size)

        self.vae.compile(optimizer='adam')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(NBC_ROOT + 'tmp/autoencoder/tmp.h5', save_best_only=True, verbose=1)
        ]
        self.vae.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)
        self.vae(self.x['train'])
        self.vae.load_weights(NBC_ROOT + 'tmp/autoencoder/tmp.h5')
        self.encodings = {}
        self.reconstructions = {}
        for type in ['train', 'dev', 'test']:
            z, _ = self.vae.encode(self.x[type])
            x_ = self.vae(self.x[type])
            self.encodings[type] = z.numpy()
            self.reconstructions[type] = x_.numpy()

    def get_autoencoder(self):
        _, seq_len, input_dim = self.x['train'].shape
        self.vae = VAE(seq_len, input_dim, self.args.vae_hidden_size, self.args.vae_beta, self.args.vae_warm_up_iters)
        if self.try_load_cached():
            return
        self.train_autoencoder()
        self.cache()
