import json
import argparse
from autoencoder import AutoencoderWrapper
import sys
sys.path.append('C:/Users/dylan/Documents')
from nbc.nbc_wrapper import NBCWrapper
import numpy as np

nbc_wrapper = None
autoencoder_wrapper = None

class Args:
    def __init__(self):
        return

def serialize(args, fname):
    if '/' in fname or '\\' in fname:
        fpath = fname
    else:
        fpath = 'config/{}.json'.format(fname)
    with open(fpath, 'w+') as f:
        f.write(json.dumps(vars(args), indent=4))

def deserialize(fname):
    if '/' in fname or '\\' in fname:
        fpath = fname
    else:
        fpath = 'config/{}.json'.format(fname)
    with open(fpath) as f:
        args_dict = json.load(f)
    args = Args()
    args.nbc_subsample = args_dict['nbc_subsample']
    args.nbc_dynamic_only = args_dict['nbc_dynamic_only']
    args.nbc_train_sequencing = args_dict['nbc_train_sequencing']
    args.nbc_dev_sequencing = args_dict['nbc_dev_sequencing']
    args.nbc_test_sequencing = args_dict['nbc_test_sequencing']
    args.nbc_chunk_size = args_dict['nbc_chunk_size']
    args.nbc_sliding_chunk_stride = args_dict['nbc_sliding_chunk_stride']
    args.nbc_label_method = args_dict['nbc_label_method']
    args.nbc_features = args_dict['nbc_features']
    args.nbc_output_type = args_dict['nbc_output_type']
    args.nbc_preprocessing = args_dict['nbc_preprocessing']
    args.vae_hidden_size = args_dict['vae_hidden_size']
    args.vae_batch_size = args_dict['vae_batch_size']
    args.vae_beta = args_dict['vae_beta']
    return args

def initialize(args):
    global nbc_wrapper
    global autoencoder_wrapper
    nbc_wrapper = NBCWrapper(args)
    autoencoder_wrapper = AutoencoderWrapper(args, nbc_wrapper)

def get_encodings(args, type='train'):
    global autoencoder_wrapper
    assert autoencoder_wrapper is not None
    z = autoencoder_wrapper.encodings[type]
    return z

def get_reconstruction(args, type='train'):
    global autoencoder_wrapper
    assert autoencoder_wrapper is not None
    x = autoencoder_wrapper.x[type]
    x_ = autoencoder_wrapper.reconstructions[type]
    return x, x_

def chunks_to_sessions(z, steps):
    sessions = {}
    for i, (key, steps_) in enumerate(steps):
        session = key[0]
        feat = z[i]
        if session not in sessions:
            sessions[session] = {'feat': [], 'steps': []}
        sessions[session]['feat'].append(feat)
        sessions[session]['steps'].append(steps_[0])
    features, steps = [], []
    for session in sessions.keys():
        steps_ = np.array(sessions[session]['steps'], dtype=int)
        feat = np.array(sessions[session]['feat'], dtype=np.float32)
        features.append(feat)
        steps.append(steps_)
    return features, steps

def get_hsmm_sequences(args):
    nbc_wrapper = NBCWrapper(args)
    autoencoder_wrapper = AutoencoderWrapper(args, nbc_wrapper)
    sequences = {}
    for type in ['train', 'dev', 'test']:
        z = autoencoder_wrapper.encodings[type]
        steps = list(nbc_wrapper.nbc.steps[type].items())
        feat, steps = chunks_to_sessions(z, steps)
        sequences[type] = (feat, steps)
    return sequences

if __name__ == '__main__':
    args = deserialize('vae8_nokl_actions')
    get_hsmm_sequences(args)
