import numpy as np
import tensorflow as tf
import sys
sys.path.append('C:/Users/dylan/Documents/')
from nbc.nbc import NBC
import argparse
import random
import scipy

random.seed(0)
tf.random.set_seed(0)

def train_classifier(x, y):
    batch_size = 10
    _, seq_len, input_dim = x['train'].shape
    n_classes = y['train'].max() + 1

    train_dset = tf.data.Dataset.from_tensor_slices((x['train'], y['train'])).batch(batch_size)
    dev_dset = tf.data.Dataset.from_tensor_slices((x['dev'], y['dev'])).batch(batch_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(seq_len, input_dim)),
        tf.keras.layers.Masking(mask_value=-1e9),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=.5)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('models/tmp.h5', save_best_only=True, verbose=1)
    ]
    model.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)

if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.subsample = 9
            self.dynamic_only = True
            self.train_sequencing = 'actions'
            self.dev_sequencing = 'actions'
            self.test_sequencing = 'actions'
            self.label_method = 'hand_motion_rhand'
            self.features = ['velY:RightHand', 'relVelZ:RightHand']
    args = Args()
    nbc = NBC(args)
    x, y = get_dset(nbc)
    train_classifier(x, y)
