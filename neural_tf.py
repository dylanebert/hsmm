import numpy as np
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append('C:/Users/dylan/Documents/')
from nbc.nbc import NBC
import uuid

def get_dataset(args, dset, type='train'):
    n_dim = next(iter(dset.features['train'].values())).shape[-1]
    max_len = 0
    for seq in dset.labels[type].values():
        if seq.shape[0] > max_len:
            max_len = seq.shape[0]
    seq_len = 2
    while seq_len < max_len:
        seq_len *= 2
    x = np.zeros((len(dset.labels[type]), seq_len, n_dim))
    y = np.zeros((len(dset.labels[type]), seq_len))
    mask = np.zeros((len(dset.labels[type]), seq_len))
    for i, seq in enumerate(dset.features[type].values()):
        x[i,:seq.shape[0],:] = seq
    for i, seq in enumerate(dset.labels[type].values()):
        y[i,:seq.shape[0]] = seq
        mask[i,:seq.shape[0]] = 1
    return tf.data.Dataset.from_tensor_slices((x, y))

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk).
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class NeuralNBC():
    def __init__(self, args):
        self.nbc = NBC(args)
        self.load_data()
        self.build_model()

    def try_load_cached(self):
        return

    def cache(self):
        return

    def load_data(self):
        self.train_dset = get_dataset(args, self.nbc, 'train')
        self.dev_dset = get_dataset(args, self.nbc, 'dev')
        self.test_dset = get_dataset(args, self.nbc, 'test')

    def build_model(self):
        #self.model = ConvNet(input_dim, 16, 5).to(device)
        #print(self.model)
        #self.loss_function = nn.CrossEntropyLoss(reduction='none')
        return

    def train(self):
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        savepath = 'models/{}.pt'.format(uuid.uuid1())
        best_loss = 1e9
        patience = 500
        k = 0
        for epoch in range(1000):
            loss = 0
            for batch in self.train_dset.batch(10):
                x, y_true = batch
                print(x.shape)
                y_pred = self.model(x)
                loss_ = self.loss_function(y_pred, y_true)
                loss_ = torch.mean(loss_ * mask)
                loss += loss_.item()
                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
            train_loss, train_acc = self.eval(self.train_loader)
            dev_loss, dev_acc = self.eval(self.dev_loader)
            print('epoch: {}, train_loss: {:.4f}, train_acc: {:.3f}, dev_loss: {:.4f}, dev_acc: {:.3f}'.format(epoch, train_loss, train_acc, dev_loss, dev_acc))
            if dev_loss < best_loss:
                best_loss = dev_loss
                print('saving model to {}'.format(savepath))
                torch.save(self.model, savepath)
                k = 0
            else:
                k += 1
                if k > patience:
                    print('loss didn\'t improve for {} epochs, stopping'.format(patience))
                    break

    def eval(self, dataloader):
        correct, total = (0, 0)
        loss = 0
        for batch in dataloader:
            x, y_true, mask = batch['x'], batch['y'], batch['mask']
            y_pred = self.model(x)
            loss_ = self.loss_function(y_pred, y_true)
            loss_ = torch.mean(loss_ * mask)
            loss += loss_.item()
            correct_, total_ = get_acc(y_pred, y_true)
            correct += correct_; total += total_
        return loss, correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    NBC.add_args(parser)
    args = parser.parse_args([
        '--subsample', '18',
        '--trim', '5',
        '--train_sequencing', 'session',
        '--dev_sequencing', 'session',
        '--test_sequencing', 'session',
        '--label_method', 'actions_rhand_apple',
        '--features',
            'speed:Apple',
            'relVelZ:RightHand',
            'velY:RightHand',
        '--preprocess'
    ])

    nbc = NeuralNBC(args)
    nbc.train()
