import numpy as np
import os
import sys
import json
import ast
import tensorflow as tf
import uuid
from sklearn import preprocessing
from sklearn.decomposition import PCA

assert 'HSMM_ROOT' in os.environ, 'set HSMM_ROOT'
assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
HSMM_ROOT = os.environ['HSMM_ROOT']
NBC_ROOT = os.environ['NBC_ROOT']
sys.path.append(NBC_ROOT)
from nbc import NBC, obj_names
from autoencoder import VAE

'''
base class for hsmm input
required attributes:
    z: features to give to the hsmm
    steps: which nbc steps the data correspond to
    lengths: length of each sequence (for padded sequences)
'''
class InputModule:
    def __init__(self):
        return

    def load(self):
        keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
        if not os.path.exists(keyspath):
            return False
        with open(keyspath) as f:
            keys = json.load(f)

        config = serialize_configuration(self)
        if config not in keys:
            return False

        savename = keys[config]
        savepath = NBC_ROOT + 'cache/input_modules/{}.json'.format(savename)
        if not os.path.exists(savepath):
            return False

        with open(savepath) as f:
            serialized = json.load(f)
        self.steps = deserialize_steps(serialized['steps'])
        self.z = deserialize_feature(serialized['z'])
        self.lengths = deserialize_feature(serialized['lengths'])

        print('loaded {} from {}'.format(config, savepath))
        return True

    def save(self):
        keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
        if os.path.exists(keyspath):
            with open(keyspath) as f:
                keys = json.load(f)
        else:
            keys = {}

        config = serialize_configuration(self)
        if config in keys:
            savename = keys[config]
        else:
            savename = str(uuid.uuid1())
            keys[config] = savename
        savepath = NBC_ROOT + 'cache/input_modules/{}.json'.format(savename)

        serialized = {
            'z': serialize_feature(self.z),
            'lengths': serialize_feature(self.lengths),
            'steps': serialize_steps(self.steps)
        }
        with open(savepath, 'w+') as f:
            json.dump(serialized, f)
        with open(keyspath, 'w+') as f:
            json.dump(keys, f)
        print('saved {} to {}'.format(config, savepath))

    def save_config(self, fname):
        self.save()
        fpath = NBC_ROOT + 'config/{}.json'.format(fname)
        with open(fpath, 'w+') as f:
            f.write(serialize_configuration(self))
        print('saved config to {}'.format(fpath))

    @classmethod
    def load_from_config(cls, fname):
        keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
        if not os.path.exists(keyspath):
            assert False, '{} does not exist'.format(keyspath)
            return None
        with open(keyspath) as f:
            keys = json.load(f)

        fpath = NBC_ROOT + 'config/{}.json'.format(fname)
        with open(fpath) as f:
            config = f.read()
        if config not in keys:
            assert False, '{} config not in keys'.format(fpath)
            return None

        savename = keys[config]
        savepath = NBC_ROOT + 'cache/input_modules/{}.json'.format(savename)
        if not os.path.exists(savepath):
            assert False, 'savepath {} does not exist'.format(savepath)
            return False

        with open(savepath) as f:
            serialized = json.load(f)
        module = InputModule()
        module.steps = deserialize_steps(serialized['steps'])
        module.z = deserialize_feature(serialized['z'])
        module.lengths = deserialize_feature(serialized['lengths'])
        print('loaded from config {}'.format(fpath))
        return module

    @classmethod
    def build_from_config(cls, fname):
        fpath = NBC_ROOT + 'config/{}.json'.format(fname)
        with open(fpath) as f:
            config = json.load(f)
        module = deserialize_configuration(config)
        print('built module from config {}'.format(fpath))
        return module

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
    def __init__(self, conditional, child):
        self.conditional = conditional
        self.child = child
        z = {'train': [], 'dev': [], 'test': []}
        lengths = {'train': [], 'dev': [], 'test': []}
        steps = {'train': {}, 'dev': {}, 'test': {}}
        for type in ['train', 'dev', 'test']:
            keys = list(conditional.steps[type].keys())
            for i, key in enumerate(keys):
                x = conditional.z[type][i]
                if not np.all(x == 0):
                    z[type].append(child.z[type][i])
                    lengths[type].append(child.lengths[type][i])
                    steps[type][key] = child.steps[type][key]
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
        if self.load():
            return
        train_dset = tf.data.Dataset.from_tensor_slices(train_module.z['train']).batch(16)
        dev_dset = tf.data.Dataset.from_tensor_slices(train_module.z['dev']).batch(16)
        _, seq_len, input_dim = train_module.z['train'].shape
        self.vae = VAE(seq_len, input_dim, 8, 1, 10000)
        self.vae.compile(optimizer='adam')
        tmp_path = NBC_ROOT + 'cache/tmp/{}.h5'.format(str(uuid.uuid1()))
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
        keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
        with open(keyspath) as f:
            keys = json.load(f)
        config = serialize_configuration(self)
        savename = keys[config]
        weightspath = NBC_ROOT + 'cache/input_modules/{}_weights.json'.format(savename)
        self.vae.save_weights(weightspath)

    def load(self, load_model=False):
        res = super().load()
        if res:
            if load_model:
                _, seq_len, input_dim = self.inference_module.z['train'].shape
                self.vae = VAE(seq_len, input_dim, 8, 1, 10000)
                self.vae(self.inference_module.z['train'][:10])
                keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
                with open(keyspath) as f:
                    keys = json.load(f)
                config = serialize_configuration(self)
                savename = keys[config]
                weightspath = NBC_ROOT + 'cache/input_modules/{}_weights.json'.format(savename)
                self.vae.load_weights(weightspath)
            return True
        return False

'''
decorator
---
convert data to session sequences
'''
class ConvertToSessions(InputModule):
    def __init__(self, child):
        self.child = child
        sessions = {'train': {}, 'dev': {}, 'test': {}}
        for type in ['train', 'dev', 'test']:
            for i, (key, steps) in enumerate(child.steps[type].items()):
                session = key[0]
                if session not in sessions[type]:
                    sessions[type][session] = {'z': [], 'steps': [], 'indices': []}
                sessions[type][session]['z'].append(child.z[type][i])
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
    def __init__(self, child):
        self.child = child
        self.z = {'train': [], 'dev': [], 'test': []}
        self.steps = {'train': {}, 'dev': {}, 'test': {}}
        self.lengths = {'train': [], 'dev': [], 'test': []}
        for type in ['train', 'dev', 'test']:
            for i, key in enumerate(child.steps[type].keys()):
                self.steps[type][key] = []
                for j in range(int(child.lengths[type][i])):
                    z = child.z[type][i,j,:]
                    steps = child.steps[type][key][j]
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
intended for 2d outputs from the autoencoder
'''
class StandardScale(InputModule):
    def __init__(self, child):
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
        self.child = child
        self.z = {}
        for type in ['train', 'dev', 'test']:
            self.z[type] = np.clip(child.z[type], -3., 3.)
        self.steps = child.steps
        self.lengths = child.lengths

'''
decorator
---
scale child to range [0, 1]
intended for 3d inputs to the autoencoder
'''
class MinMax(InputModule):
    def __init__(self, child):
        self.child = child
        self.z = {}
        scalers = []
        for i in range(child.z['train'].shape[1]):
            scaler = preprocessing.MinMaxScaler().fit(child.z['train'][:,i,:])
            scalers.append(scaler)
        for type in ['train', 'dev', 'test']:
            self.z[type] = np.zeros(child.z[type].shape)
            for i in range(child.z['train'].shape[1]):
                self.z[type][:,i,:] = np.clip(scalers[i].transform(child.z[type][:,i,:]), 0., 1.)
        self.steps = child.steps
        self.lengths = child.lengths

'''
decorator
---
use pca to reduce child dim to 2
'''
class ReducePCA(InputModule):
    def __init__(self, child):
        self.child = child
        self.z = {}
        scaler = PCA().fit(child.z['train'])
        for type in ['train', 'dev', 'test']:
            self.z[type] = scaler.transform(child.z[type])
        self.steps = child.steps
        self.lengths = child.lengths

'''
composite
---
concatenate children
'''
class Concat(InputModule):
    def __init__(self, children):
        self.children = children
        if self.load():
            return
        self.z = {'train': [], 'dev': [], 'test': []}
        for type in ['train', 'dev', 'test']:
            for child in children:
                self.z[type].append(child.z[type])
            self.z[type] = np.concatenate(self.z[type], axis=-1)
        self.steps = children[0].steps
        self.lengths = children[0].lengths
        self.save()

'''
composite
---
get max at magnitude at each index
'''
class Max(InputModule):
    def __init__(self, conditionals, children, add_indices):
        self.conditionals = conditionals
        self.children = children
        self.add_indices = add_indices
        if self.load():
            return

        z_cond = {}
        z = {}
        assert len(conditionals) == len(children)
        for type in ['train', 'dev', 'test']:
            z_cond_ = []
            z_ = []
            for i in range(len(conditionals)):
                z_cond_.append(conditionals[i].z[type])
                z_.append(children[i].z[type])
            z_cond_ = np.stack(z_cond_, axis=1)
            z_ = np.stack(z_, axis=1)
            z_cond[type] = z_cond_
            z[type] = z_

        self.z = {}
        self.max_indices = {}
        self.steps = children[0].steps
        self.lengths = children[0].lengths
        for type in ['train', 'dev', 'test']:
            magnitudes = np.linalg.norm(z_cond[type], axis=(2, 3))
            max_indices = np.argmax(magnitudes, axis=1)
            print(np.unique(max_indices, return_counts=True)[1])

            max_z = []
            max_steps = []
            max_lengths = []
            for i in range(z[type].shape[0]):
                max_z.append(z[type][i, max_indices[i], :])
            max_z = np.array(max_z)

            if add_indices:
                one_hot = []
                n_objs = len(conditionals)
                for i in range(z[type].shape[0]):
                    v = magnitudes[i, max_indices[i]]
                    vec = np.zeros((n_objs,))
                    if v > 0.:
                        vec[max_indices[i]] = 1
                    one_hot.append(vec)
                one_hot = np.array(one_hot)
                max_z = np.concatenate((max_z, one_hot), axis=-1)

            self.z[type] = max_z
            self.max_indices[type] = max_indices
        self.save()

    def save(self):
        super().save()
        keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
        with open(keyspath) as f:
            keys = json.load(f)
        config = serialize_configuration(self)
        savename = keys[config]
        indicespath = NBC_ROOT + 'cache/input_modules/{}_indices.json'.format(savename)
        with open(indicespath, 'w+') as f:
            json.dump(serialize_feature(self.max_indices), f)

    def load(self):
        res = super().load()
        if res:
            keyspath = NBC_ROOT + 'cache/input_modules/keys.json'
            with open(keyspath) as f:
                keys = json.load(f)
            config = serialize_configuration(self)
            savename = keys[config]
            indicespath = NBC_ROOT + 'cache/input_modules/{}_indices.json'.format(savename)
            with open(indicespath) as f:
                self.max_indices = deserialize_feature(json.load(f))
            return True
        return False

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
            try:
                key_tuple = ast.literal_eval(key)
            except:
                key_tuple = (key,)
            steps[type][key_tuple] = np.array(serialized[type][key])
    return steps
def serialize_configuration(module):
    #leaves
    if isinstance(module, NBCChunks):
        return json.dumps({'type': 'NBCChunks', 'obj': module.obj}, indent=4)
    if isinstance(module, DirectInputModule):
        return json.dumps({'type': 'DirectInputModule', 'obj': module.obj}, indent=4)

    #decorators
    if isinstance(module, Trim):
        conditional_config = serialize_configuration(module.conditional)
        child_config = serialize_configuration(module.child)
        return json.dumps({'type': 'Trim', 'conditional': conditional_config, 'child': child_config}, indent=4)
    if isinstance(module, Autoencoder):
        train_config = serialize_configuration(module.train_module)
        inference_config = serialize_configuration(module.inference_module)
        return json.dumps({'type': 'Autoencoder', 'train_config': train_config, 'inference_config': inference_config})
    if isinstance(module, ConvertToSessions):
        return json.dumps({'type': 'ConvertToSessions', 'child': serialize_configuration(module.child)})
    if isinstance(module, ConvertToChunks):
        return json.dumps({'type': 'ConvertToChunks', 'child': serialize_configuration(module.child)})
    if isinstance(module, StandardScale):
        return json.dumps({'type': 'StandardScale', 'child': serialize_configuration(module.child)})
    if isinstance(module, Clip):
        return json.dumps({'type': 'Clip', 'child': serialize_configuration(module.child)})
    if isinstance(module, MinMax):
        return json.dumps({'type': 'MinMax', 'child': serialize_configuration(module.child)})
    if isinstance(module, ReducePCA):
        return json.dumps({'type': 'ReducePCA', 'child': serialize_configuration(module.child)})

    #composite
    if isinstance(module, Concat):
        children = [serialize_configuration(child) for child in module.children]
        return json.dumps({'type': 'Concat', 'children': children})
    if isinstance(module, Max):
        conditionals = [serialize_configuration(conditional) for conditional in module.conditionals]
        children = [serialize_configuration(child) for child in module.children]
        return json.dumps({'type': 'Max', 'conditionals': conditionals, 'children': children, 'add_indices': module.add_indices})
def deserialize_configuration(config):
    if isinstance(config, str):
        config = json.loads(config)

    #leaves
    if config['type'] == 'NBCChunks':
        return NBCChunks(config['obj'])
    if config['type'] == 'DirectInputconfig':
        return DirectInputconfig(config['obj'])

    #decorators
    if config['type'] == 'Trim':
        conditional = deserialize_configuration(config['conditional'])
        child = deserialize_configuration(config['child'])
        return Trim(conditional, child)
    if config['type'] == 'Autoencoder':
        train_config = deserialize_configuration(config['train_config'])
        inference_config = deserialize_configuration(config['inference_config'])
        return Autoencoder(train_config, inference_config)
    if config['type'] == 'ConvertToSessions':
        return ConvertToSessions(deserialize_configuration(config['child']))
    if config['type'] == 'ConvertToChunks':
        return ConvertToChunks(deserialize_configuration(config['child']))
    if config['type'] == 'StandardScale':
        return StandardScale(deserialize_configuration(config['child']))
    if config['type'] == 'Clip':
        return Clip(deserialize_configuration(config['child']))
    if config['type'] == 'MinMax':
        return MinMax(deserialize_configuration(config['child']))
    if config['type'] == 'ReducePCA':
        return ReducePCA(deserialize_configuration(config['child']))

    #composite
    if config['type'] == 'Concat':
        children = [deserialize_configuration(child) for child in config['children']]
        return Concat(children)
    if config['type'] == 'Max':
        conditionals = [deserialize_configuration(conditional) for conditional in config['conditionals']]
        children = [deserialize_configuration(child) for child in config['children']]
        return Max(conditionals, children, config['add_indices'])
def report(module):
    print(module.z['dev'].shape)
    print(np.mean(module.z['dev']))
    print(np.std(module.z['dev']))

if __name__ == '__main__':
    '''obj = sys.argv[1]
    data = NBCChunks(obj)
    preprocessed = MinMax(Clip(data))
    trimmed = Trim(data, preprocessed)
    autoencoder = Autoencoder(trimmed, preprocessed)
    output = ConvertToSessions(StandardScale(autoencoder))
    output.save_config('autoencoder_{}'.format(obj))'''

    autoencoders = []
    conditionals = []
    for obj in obj_names:
        conditional = NBCChunks(obj)
        autoencoder = InputModule.build_from_config('autoencoder_{}'.format(obj)).child #child is standard scaled autoencoder encodings
        conditionals.append(conditional)
        autoencoders.append(autoencoder)
    combined = Max(conditionals, autoencoders, True)
    output = ConvertToSessions(combined)
    output.save_config('max_objs_indices')
