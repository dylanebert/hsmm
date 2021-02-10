import numpy as np
import tensorflow as tf

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
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim, activation='sigmoid'))
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
        warmup_coef = tf.math.minimum(tf.cast(self.optimizer.iterations, tf.float32) / self.warm_up_iters, tf.cast(1., tf.float32)) ** 3.
        beta = self.beta * warmup_coef
        with tf.GradientTape() as tape:
            reconstr_loss, kl_loss = self.compute_loss(x)
            loss = reconstr_loss + beta * kl_loss
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
        reconstr_loss, kl_loss = self.compute_loss(x)
        loss = reconstr_loss + self.beta * kl_loss
        self.dev_loss(loss)
        return {
            'loss': self.dev_loss.result()
        }
