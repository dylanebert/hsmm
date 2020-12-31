import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append('C:/Users/dylan/Documents/')
from nbc.nbc import NBC
import uuid
import json
import os
import matplotlib.pyplot as plt
from scipy import stats
import random

random.seed(0)
tf.random.set_seed(0)

class Autoencoder(tf.keras.models.Model):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(seq_len, input_dim)),
            tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(4, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_dim)
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(seq_len),
            tf.keras.layers.Reshape((seq_len // 4, 4)),
            tf.keras.layers.Conv1DTranspose(4, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1DTranspose(8, 3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1DTranspose(input_dim, 3, padding='same')
        ])

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

def cache(args, id):
    args_dict = json.dumps(vars(args))
    key_path = 'models/keys.json'
    if os.path.exists(key_path):
        with open(key_path) as f:
            keys = json.load(f)
    else:
        keys = {}
    keys[args_dict] = id
    with open(key_path, 'w+') as f:
        json.dump(keys, f)
    print('cached model')

def try_load_cached(args):
    args_dict = json.dumps(vars(args))
    key_path = 'models/keys.json'
    if not os.path.exists(key_path):
        return False
    with open(key_path) as f:
        keys = json.load(f)
    if args_dict not in keys:
        return False
    fpath = 'models/{}.h5'.format(keys[args_dict])
    if os.path.exists(fpath):
        return fpath
    else:
        del keys[args_dict]
        with open(key_path, 'w+') as f:
            json.dump(keys, f)
        return False

def get_autoencoder(args, x):
    batch_size = 10
    _, seq_len, input_dim = x['train'].shape
    hidden_dim = args.hidden_size

    train_dset = tf.data.Dataset.from_tensor_slices((x['train'], x['train'])).batch(batch_size, drop_remainder=True)
    dev_dset = tf.data.Dataset.from_tensor_slices((x['dev'], x['dev'])).batch(batch_size, drop_remainder=True)

    model = Autoencoder(seq_len, input_dim, hidden_dim)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.build(input_shape=(batch_size, seq_len, input_dim))
    model.encoder.summary()
    model.decoder.summary()

    fpath = try_load_cached(args)
    if fpath:
        model.load_weights(fpath)
    else:
        id = str(uuid.uuid1())
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint('models/{}.h5'.format(id), save_best_only=True, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='logs/')
        ]
        model.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks)
        cache(args, id)

    return model

def classifier_test(args, model, x, y):
    z = {}
    for type in ['train', 'dev', 'test']:
        z[type] = model.encoder(x[type])
    batch_size = 10
    num_classes = y['train'].max() + 1
    print('num classes', num_classes)
    classifiers = {
        'original': tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(model.seq_len, model.input_dim)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(num_classes)
        ]),
        'encoded': tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(model.hidden_dim)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(num_classes)
        ])
    }
    train_dset = {
        'original': tf.data.Dataset.from_tensor_slices((x['train'], y['train'])).batch(batch_size, drop_remainder=True),
        'encoded': tf.data.Dataset.from_tensor_slices((z['train'], y['train'])).batch(batch_size, drop_remainder=True)
    }
    dev_dset = {
        'original': tf.data.Dataset.from_tensor_slices((x['dev'], y['dev'])).batch(batch_size, drop_remainder=True),
        'encoded': tf.data.Dataset.from_tensor_slices((z['dev'], y['dev'])).batch(batch_size, drop_remainder=True)
    }
    for name, classifier in classifiers.items():
        print(name)
        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, verbose=0),
            tf.keras.callbacks.ModelCheckpoint('models/tmp.h5', save_best_only=True, verbose=0)
        ]
        classifier.fit(x=train_dset[name], epochs=1000, shuffle=True, validation_data=dev_dset[name], callbacks=callbacks, verbose=0)
        classifier.load_weights('models/tmp.h5')
        classifier.evaluate(x=train_dset[name])

def summarize_dset(dset):
    for type in ['train', 'dev', 'test']:
        print(type)
        print(pd.DataFrame({'y': dset[type]}).groupby('y').size())

def sliding_window_dset(nbc, chunk_size=12, stride=4, include_idle=False):
    x = {}; y = {}
    for type in ['train', 'dev', 'test']:
        x_ = []; y_ = []
        for key, seq in nbc.labels[type].items():
            n = seq.shape[0]
            k = 0
            while k < n - chunk_size:
                chunk = seq[k:k+chunk_size]
                if (include_idle or chunk[0] > 1) and np.all(chunk == chunk[0]):
                    label = int(chunk[0])
                    feat = nbc.features[type][key][k:k+chunk_size]
                    if include_idle:
                        label -= 1
                    else:
                        label -= 2
                    x_.append(feat)
                    y_.append(label)
                    k += stride
                else:
                    k += 1
        x[type] = np.array(x_, dtype=np.float32)
        y[type] = np.array(y_, dtype=int)
    return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, required=True)
    NBC.add_args(parser)
    '''args = parser.parse_args([
        '--subsample', '5',
        '--train_sequencing', 'session',
        '--dev_sequencing', 'session',
        '--test_sequencing', 'session',
        '--label_method', 'actions',
        '--features',
            'relPosX:hands',
            'posY:hands',
            'relPosZ:hands',
            'dist_to_rhand:objs',
            'speed:objs',
        '--preprocess',
        '--hidden_size', '10'
    ])'''
    args = parser.parse_args([
        '--subsample', '5',
        '--train_sequencing', 'session',
        '--dev_sequencing', 'session',
        '--test_sequencing', 'session',
        '--label_method', 'actions_rhand_apple',
        '--features',
            'relPosX:RightHand',
            'posY:RightHand',
            'relPosZ:RightHand',
            'dist_to_rhand:Apple',
            'speed:Apple',
        '--preprocess',
        '--hidden_size', '10'
    ])

    nbc = NBC(args)
    x, y = sliding_window_dset(nbc)
    summarize_dset(y)
    model = get_autoencoder(args, x)
    classifier_test(args, model, x, y)
