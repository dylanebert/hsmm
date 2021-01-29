from autoencoder import AutoencoderWrapper
import sys
sys.path.append('C:/Users/dylan/Documents')
from nbc.nbc_wrapper import NBCWrapper
import numpy as np
import config

nbc_wrapper = None
autoencoder_wrapper = None

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
    args = config.deserialize('vae8')
    initialize(args)
    z = get_encodings(args)
    print(z)
