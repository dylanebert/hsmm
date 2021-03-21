import numpy as np
import os
import sys
import json
import ast
import tensorflow as tf
import uuid
from sklearn import preprocessing

assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
from nbc import NBC
from autoencoder import VAE

'''
base class for hsmm input
required attributes:
    z: features to give to the hsmm
    steps: which nbc steps the data correspond to
    lengths: length of each sequence (for padded sequences)
'''
class InputModule:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(NBC_ROOT + 'config/{}.json'.format(config_path)) as f:
            config = json.load(f)
        module = expand(config)
        self.z = module.z
        self.steps = module.steps
        self.lengths = module.lengths

    def load(self):
        savepath = NBC_ROOT + 'cache/input_modules/{}.json'.format(self.fname)
        if not os.path.exists(savepath):
            return False
        with open(savepath) as f:
            serialized = json.load(f)
        self.steps = deserialize_steps(serialized['steps'])
        self.z = deserialize_feature(serialized['z'])
        self.lengths = deserialize_feature(serialized['lengths'])
        print('loaded {}'.format(savepath))
        return True

    def save(self):
        savepath = NBC_ROOT + 'cache/input_modules/{}.json'.format(self.fname)
        serialized = {
            'z': serialize_feature(self.z),
            'lengths': serialize_feature(self.lengths),
            'steps': serialize_steps(self.steps)
        }
        with open(savepath, 'w+') as f:
            json.dump(serialized, f)
        print('saved to {}'.format(savepath))

'''
leaf
---
features directly from nbc to hsmm, at every timestep
'''
class DirectInputModule(InputModule):
    @classmethod
    def default_args(cls):
        return {
            'nbc_subsample': 9,
            'nbc_dynamic_only': True,
            'nbc_train_sequencing': 'session',
            'nbc_dev_sequencing': 'session',
            'nbc_test_sequencing': 'session',
            'nbc_chunk_size': 10,
            'nbc_sliding_chunk_stride': 3,
            'nbc_label_method': 'none',
            'nbc_features': [],
            'nbc_output_type': 'classifier',
            'nbc_preprocessing': ['clip', 'tanh']
        }

    def __init__(self, obj):
        self.obj = obj
        self.fname = 'direct_inputs_{}'.format(obj)
        if self.load():
            return
        args = Args()
        for k, v in DirectInputModule.default_args().items():
            setattr(args, k, v)
        feat = ['{}{}'.format(param, obj) for param in ['relVelX:', 'velY:', 'relVelZ:']]
        setattr(args, 'nbc_features', feat)
        nbc = NBC(args)
        seq_len = 0
        n_dim = next(iter(nbc.features['train'].values())).shape[-1]
        for type in ['train', 'dev', 'test']:
            for feat in nbc.features[type].values():
                if feat.shape[0] > seq_len:
                    seq_len = feat.shape[0]
        z = {}
        lengths = {}
        for type in ['train', 'dev', 'test']:
            z_ = np.zeros((len(nbc.features[type]), seq_len, n_dim))
            lengths_ = np.zeros((len(nbc.features[type]),)).astype(int)
            for i, feat in enumerate(nbc.features[type].values()):
                z_[i, :feat.shape[0], :] = feat
                lengths_[i] = feat.shape[0]
            z[type] = z_
            lengths[type] = lengths_
        self.z = z
        self.lengths = lengths
        self.steps = nbc.steps
        self.save()

'''
leaf
---
wrapper for nbc sliding chunks
'''
class NBCChunks(InputModule):
    @classmethod
    def default_args(cls):
        return {
            'nbc_subsample': 9,
            'nbc_dynamic_only': True,
            'nbc_train_sequencing': 'sliding_chunks',
            'nbc_dev_sequencing': 'sliding_chunks',
            'nbc_test_sequencing': 'sliding_chunks',
            'nbc_chunk_size': 10,
            'nbc_sliding_chunk_stride': 3,
            'nbc_label_method': 'none',
            'nbc_features': [],
            'nbc_output_type': 'classifier',
            'nbc_preprocessing': ['clip', 'tanh']
        }

    def __init__(self, obj):
        self.obj = obj
        self.fname = 'nbc_chunks_{}'.format(obj)
        if self.load():
            return
        args = Args()
        for k, v in NBCChunks.default_args().items():
            setattr(args, k, v)
        feat = ['{}{}'.format(param, obj) for param in ['relVelX:', 'velY:', 'relVelZ:']]
        setattr(args, 'nbc_features', feat)
        nbc = NBC(args)
        z = {}
        lengths = {}
        for type in ['train', 'dev', 'test']:
            z[type] = np.stack(list(nbc.features[type].values()), axis=0).astype(np.float32)
            lengths[type] = (np.ones((len(z[type],))) * z[type].shape[1]).astype(int)
        self.z = z
        self.lengths = lengths
        self.steps = nbc.steps
        self.save()

'''
decorator
---
trim zeros from child
'''
class Trim(InputModule):
    def __init__(self, input_module):
        self.input_module = input_module
        self.fname = '{}_trim'.format(input_module.fname)
        z = {'train': [], 'dev': [], 'test': []}
        lengths = {'train': [], 'dev': [], 'test': []}
        steps = {'train': {}, 'dev': {}, 'test': {}}
        for type in ['train', 'dev', 'test']:
            keys = list(input_module.steps[type].keys())
            for i, key in enumerate(keys):
                x = input_module.z[type][i]
                if not np.all(x == 0):
                    z[type].append(x)
                    lengths[type].append(input_module.lengths[type][i])
                    steps[type][key] = input_module.steps[type][key]
            z[type] = np.array(z[type], dtype=np.float32)
            lengths[type] = np.array(lengths[type], dtype=int)
        self.z = z
        self.lengths = lengths
        self.steps = steps

'''
decorator
---
use vae to encode input to lower dimension
requires child for inference and child for training
'''
class Autoencoder(InputModule):
    def __init__(self, train_module, inference_module):
        self.train_module = train_module
        self.inference_module = inference_module
        self.fname = 'autoencoder_{}_{}'.format(train_module.fname, inference_module.fname)
        if self.load():
            return
        train_dset = tf.data.Dataset.from_tensor_slices(train_module.z['train']).batch(16)
        dev_dset = tf.data.Dataset.from_tensor_slices(train_module.z['dev']).batch(16)
        _, seq_len, input_dim = train_module.z['train'].shape
        self.vae = VAE(seq_len, input_dim, 8, 1, 10000)
        self.vae.compile(optimizer='adam')
        tmp_path = NBC_ROOT + 'cache/autoencoder/tmp_{}.h5'.format(str(uuid.uuid1()))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=25, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(tmp_path, save_best_only=True, verbose=1)
        ]
        self.vae.fit(x=train_dset, epochs=1000, shuffle=True, validation_data=dev_dset, callbacks=callbacks, verbose=1)
        self.vae(train_module.z['train'][:10])
        self.vae.load_weights(tmp_path)

        self.z = {}
        self.lengths = {}
        self.reconstructions = {}
        for type in ['train', 'dev', 'test']:
            z = self.vae.encode(inference_module.z[type])[0].numpy()
            x_ = self.vae(inference_module.z[type]).numpy()
            self.z[type] = z
            self.lengths[type] = np.ones((z.shape[0],))
            self.reconstructions[type] = x_
        self.steps = inference_module.steps
        self.save()

    def save(self):
        super().save()
        weightspath = NBC_ROOT + 'cache/input_modules/{}_weights.json'.format(self.fname)
        self.vae.save_weights(weightspath)

    def load(self, load_model=False):
        res = super().load()
        if res:
            if load_model:
                _, seq_len, input_dim = self.inference_module.z['train'].shape
                self.vae = VAE(seq_len, input_dim, 8, 1, 10000)
                self.vae(self.inference_module.z['train'][:10])
                weightspath = NBC_ROOT + 'cache/input_modules/{}_weights.json'.format(self.fname)
                self.vae.load_weights(weightspath)
            return True
        return False

'''
decorator
---
convert data to session sequences
'''
class ConvertToSessions(InputModule):
    def __init__(self, input_module):
        self.fname = '{}_to_sessions'.format(input_module.fname)
        self.child = input_module
        sessions = {'train': {}, 'dev': {}, 'test': {}}
        for type in ['train', 'dev', 'test']:
            for i, (key, steps) in enumerate(input_module.steps[type].items()):
                session = key[0]
                if session not in sessions[type]:
                    sessions[type][session] = {'z': [], 'steps': [], 'indices': []}
                sessions[type][session]['z'].append(input_module.z[type][i])
                sessions[type][session]['steps'].append(steps)
                sessions[type][session]['indices'].append(i)
        n_dim = next(iter(sessions['train'].values()))['z'][0].shape[-1]
        seq_len = 0
        for type in ['train', 'dev', 'test']:
            for session in sessions[type].keys():
                n = len(sessions[type][session]['z'])
                if n > seq_len:
                    seq_len = n
        self.z = {}
        self.steps = {}
        self.lengths = {}
        for type in ['train', 'dev', 'test']:
            n = len(sessions[type].keys())
            self.z[type] = np.zeros((n, seq_len, n_dim))
            self.lengths[type] = np.zeros((n,))
            self.steps[type] = {}
            for i, session in enumerate(sessions[type].keys()):
                z = np.array(sessions[type][session]['z'], dtype=np.float32)
                steps = np.array(sessions[type][session]['steps'], dtype=int)
                self.z[type][i, :z.shape[0], :] = z
                self.lengths[type][i] = z.shape[0]
                self.steps[type][session] = steps

'''
decorator
---
convert sessions to chunks
inverse of ConvertToSessions
'''
class ConvertToChunks(InputModule):
    def __init__(self, input_module):
        self.fname = '{}_to_chunks'.format(input_module.fname)
        self.z = {'train': [], 'dev': [], 'test': []}
        self.steps = {'train': {}, 'dev': {}, 'test': {}}
        self.lengths = {'train': [], 'dev': [], 'test': []}
        for type in ['train', 'dev', 'test']:
            for i, key in enumerate(input_module.steps[type].keys()):
                self.steps[type][key] = []
                for j in range(int(input_module.lengths[type][i])):
                    z = input_module.z[type][i,j,:]
                    steps = input_module.steps[type][key][j]
                    self.z[type].append(z)
                    self.lengths[type].append(1)
                    self.steps[type][key].append(steps)
                self.steps[type][key] = np.array(self.steps[type][key], dtype=int)
            self.z[type] = np.array(self.z[type], dtype=np.float32)
            self.lengths[type] = np.array(self.lengths[type], dtype=int)

'''
decorator
---
standard scale inputs
'''
class StandardScale(InputModule):
    def __init__(self, child):
        self.fname = '{}_scaled'.format(child.fname)
        self.child = child
        self.z = {}
        scaler = preprocessing.StandardScaler().fit(child.z['train'])
        for type in ['train', 'dev', 'test']:
            self.z[type] = scaler.transform(child.z[type])
        self.steps = child.steps
        self.lengths = child.lengths

'''
decorator
---
clip child to range [-3, 3]
'''
class Clip(InputModule):
    def __init__(self, child):
        self.fname = '{}_clip'.format(child.fname)
        self.child = child
        self.z = {}
        for type in ['train', 'dev', 'test']:
            self.z[type] = np.clip(child.z[type], -3., 3.)
        self.steps = child.steps
        self.lengths = child.lengths

'''
decorator
---
apply tanh estimator scaling to child
'''
class Tanh(InputModule):
    def __init__(self, child):
        self.fname = '{}_tanh'.format(child.fname)
        self.child = child
        m = np.mean(child.z['train'], axis=0)
        std = np.std(child.z['train'], axis=0)
        self.z = {}
        for type in ['train', 'dev', 'test']:
            self.z[type] = .5 * (np.tanh(.01 * ((child.z[type] - m) / std)) + 1)
        self.steps = child.steps
        self.lengths = child.lengths

#-----------------------utility functions-----------------------
class Args:
    def __init__(self):
        return
def serialize_feature(x):
    serialized = {}
    for type in ['train', 'dev', 'test']:
        serialized[type] = x[type].tolist()
    return serialized
def deserialize_feature(serialized):
    x = {}
    for type in ['train', 'dev', 'test']:
        x[type] = np.array(serialized[type])
    return x
def serialize_steps(steps):
    serialized = {'train': {}, 'dev': {}, 'test': {}}
    for type in ['train', 'dev', 'test']:
        for key in steps[type].keys():
            serialized[type][str(key)] = steps[type][key].tolist()
    return serialized
def deserialize_steps(serialized):
    steps = {'train': {}, 'dev': {}, 'test': {}}
    for type in ['train', 'dev', 'test']:
        for key in serialized[type].keys():
            key_tuple = ast.literal_eval(key)
            steps[type][key_tuple] = np.array(serialized[type][key])
    return steps
def expand(module_config):
    #leaves
    if module_config['type'] == 'NBCChunks':
        return NBCChunks(module_config['obj'])
    if module_config['type'] == 'DirectInputModule':
        return DirectInputModule(module_config['obj'])

    #decorators
    if module_config['type'] == 'Trim':
        return Trim(expand(module_config['child']))
    if module_config['type'] == 'Autoencoder':
        train_module = expand(module_config['train_module'])
        inference_module = expand(module_config['inference_module'])
        return Autoencoder(train_module, inference_module)
    if module_config['type'] == 'ConvertToSessions':
        return ConvertToSessions(expand(module_config['child']))
    if module_config['type'] == 'ConvertToChunks':
        return ConvertToChunks(expand(module_config['child']))
    if module_config['type'] == 'StandardScale':
        return StandardScale(expand(module_config['child']))
    if module_config['type'] == 'Clip':
        return Clip(expand(module_config['child']))
    if module_config['type'] == 'Tanh':
        return Tanh(expand(module_config['child']))

if __name__ == '__main__':
    data = InputModule(NBC_ROOT + 'config/hsmm.json')
    print(data.z['dev'].shape)
