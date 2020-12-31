import numpy as np
import tensorflow as tf
import sys
sys.path.append('C:/Users/dylan/Documents/')
from nbc.nbc import NBC
import argparse
import random
from scipy import interpolate

random.seed(0)
tf.random.set_seed(0)

def get_dset(nbc):
    x = {}; y = {}
    max_len = 0
    n_dim = next(iter(nbc.features['train'].values())).shape[-1]
    seq_len = 8
    for type in ['train', 'dev', 'test']:
        x_ = []; y_ = []
        for key, seq in nbc.labels[type].items():
            if not np.all(seq[0] == seq):
                continue
            label = int(seq[0]) - 1
            if label < 0:
                continue
            feat = nbc.features[type][key]
            _x, _y = np.arange(0, feat.shape[1]), np.arange(0, feat.shape[0])
            f = interpolate.interp2d(_x, _y, feat)
            _x_, _y_ = np.arange(0, feat.shape[1]), np.arange(0, seq_len)
            feat_ = f(_x_, _y_)
            x_.append(feat_)
            y_.append(label)
        x[type] = np.array(x_, dtype=np.float32)
        y[type] = np.array(y_, dtype=int)
    return x, y

def train_classifier(args, x, y):
    batch_size = 10
    _, seq_len, input_dim = x['train'].shape
    num_classes = y['train'].max() + 1

    train_dset = tf.data.Dataset.from_tensor_slices((x['train'], y['train'])).batch(batch_size)
    dev_dset = tf.data.Dataset.from_tensor_slices((x['dev'], y['dev'])).batch(batch_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(seq_len, input_dim)),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
        tf.keras.layers.Conv1D(16, kernel_size=3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('models/tmp.h5', save_best_only=True, verbose=1)
    ]
    model.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    NBC.add_args(parser)
    args = parser.parse_args([
        '--subsample', '5',
        '--train_sequencing', 'actions',
        '--dev_sequencing', 'actions',
        '--test_sequencing', 'actions',
        '--label_method', 'actions',
        '--features',
            'relPosX:hands',
            'posY:hands',
            'relPosZ:hands',
            'dist_to_rhand:objs',
            'dist_to_lhand:objs',
            'speed:objs',
        '--preprocess'
    ])

    nbc = NBC(args)
    x, y = get_dset(nbc)
    train_classifier(args, x, y)
