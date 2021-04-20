import numpy as np
import tensorflow as tf


class LSTM(tf.keras.models.Model):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.lstm1 = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(.5)
        self.dense = tf.keras.layers.Dense(input_dim)

    def call(self, x, training=None):
        x = self.lstm1(x)
        x = self.lstm2(x)
        if training:
            x = self.dropout(x)
        x = self.dense(x)
        return x

    def encode(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        return x


def transform_data(data, window=2, stride=1, lag=1):
    x = {'train': [], 'dev': [], 'test': []}
    y = {'train': [], 'dev': [], 'test': []}
    for type in ['train', 'dev', 'test']:
        z = data.z[type]
        lengths = data.lengths[type]
        for i in range(z.shape[0]):
            z_ = z[i]
            length = lengths[i]
            n_chunks = (length - window - lag) // stride
            for j in range(n_chunks):
                x_ = z_[j * stride:j * stride + window]
                y_ = z_[j * stride + window + lag]
                x[type].append(x_)
                y[type].append(y_)
    for type in ['train', 'dev', 'test']:
        x[type] = np.array(x[type], dtype=np.float32)
        y[type] = np.array(y[type], dtype=np.float32)
    return x, y


if __name__ == '__main__':
    import input_modules
    data = input_modules.DirectInputModule('Apple', 9)
    trimmed = input_modules.Trim(data, data)
    x, y = transform_data(trimmed, window=10, stride=1, lag=5)
    _, seq_len, input_dim = x['train'].shape
    lstm = LSTM(seq_len, input_dim, 8)
    lstm.compile(optimizer='adam', loss='mse')
    tmp_path = 'D:/nbc/cache/tmp/tmp.h5'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=25, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(tmp_path, save_best_only=True, verbose=1)
    ]
    lstm.fit(x=x['train'], y=y['train'], epochs=1000, shuffle=True, validation_data=(x['dev'], y['dev']), callbacks=callbacks, verbose=1)

    for type in ['train', 'dev', 'test']:
        z = lstm.encode(x[type]).numpy()

    '''import matplotlib.pyplot as plt
    y_pred = lstm(x['dev']).numpy()
    real = np.mean(y['dev'], axis=-1)# y['dev'][:,2]
    pred = np.mean(y_pred, axis=-1)# y_pred[:,2]
    _x = np.arange(0, real.shape[0], 1)
    plt.plot(_x, real)
    plt.plot(_x, pred)
    plt.show()'''
    # x_ = lstm(x['dev'])
