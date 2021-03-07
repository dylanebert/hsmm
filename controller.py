import numpy as np
import pandas as pd
from autoencoder_wrapper import AutoencoderWrapper
from hsmm_wrapper import HSMMWrapper
from sklearn import preprocessing
import os
import sys
import argparse
from eval import OddManOut

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
sys.path.append(os.environ['NBC_ROOT'])
from nbc import NBC
from nbc_wrapper import NBCWrapper
import config

nbc = None
nbc_wrapper = None
autoencoder_wrapper = None
hsmm_wrapper = None

def initialize(args, model):
    global nbc
    global nbc_wrapper
    global autoencoder_wrapper
    global hsmm_wrapper
    if model in ['nbc', 'autoencoder', 'hsmm', 'eval']:
        nbc_wrapper = NBCWrapper(args)
    if model in ['autoencoder', 'hsmm', 'eval']:
        autoencoder_wrapper = AutoencoderWrapper(args, nbc_wrapper)
    if model in ['hsmm', 'eval']:
        hsmm_wrapper = HSMMWrapper(args, nbc_wrapper=nbc_wrapper, autoencoder_wrapper=autoencoder_wrapper)
    if model == 'hsmm_direct':
        nbc = NBC(args)
        hsmm_wrapper = HSMMWrapper(args, nbc=nbc)

def get_encodings(args, session, type='train'):
    global autoencoder_wrapper, hsmm_wrapper
    assert autoencoder_wrapper is not None and hsmm_wrapper is not None
    indices = hsmm_wrapper.sequences[type][2]
    predictions = hsmm_wrapper.predictions[type]
    z = autoencoder_wrapper.reduced_encodings()[type]
    keys = list(nbc_wrapper.nbc.steps[type].keys())
    steps = list(nbc_wrapper.nbc.steps[type].values())
    data = []
    for i in range(len(indices)):
        session_ = keys[indices[i][0]][0]
        assert session_ == keys[indices[i][-1]][0]
        if session_ == session:
            for j, idx in enumerate(indices[i]):
                data.append({
                    'start_step': steps[idx][0],
                    'end_step': steps[idx][-1],
                    'encoding': z[idx],
                    'label': predictions[i][j],
                    'idx': idx
                })
    return pd.DataFrame(data)

def get_reconstruction(args, idx, type='train'):
    global autoencoder_wrapper
    assert autoencoder_wrapper is not None
    x = autoencoder_wrapper.x[type][idx]
    x_ = autoencoder_wrapper.reconstructions[type][idx]
    return x, x_

def get_predictions(args, session=None, type='train'):
    global hsmm_wrapper
    assert hsmm_wrapper is not None
    indices = hsmm_wrapper.sequences[type][2]
    predictions = hsmm_wrapper.predictions[type]
    if session is None:
        return predictions, indices
    else:
        sessions = get_sessions(type)
        session_idx = sessions.index(session)
        return predictions[session_idx], indices[session_idx]

def get_sessions(type='train'):
    global nbc
    global nbc_wrapper
    if nbc is not None:
        keys = list(nbc.steps[type].keys())
        sessions = [key[0] for key in keys]
    else:
        assert nbc_wrapper is not None
        keys = list(nbc_wrapper.nbc.steps[type].keys())
        sessions = np.unique([key[0] for key in keys]).tolist()
    return sessions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='hidden=16')
    parser.add_argument('--model', type=str, choices=['nbc', 'autoencoder', 'hsmm', 'eval', 'hsmm_direct'], default='autoencoder')
    cmd_args = parser.parse_args()
    args = config.deserialize(cmd_args.config)
    initialize(args, cmd_args.model)
    if cmd_args.model == 'nbc':
        print(nbc_wrapper.nbc.steps['train'].items())
    if cmd_args.model == 'autoencoder':
        z = autoencoder_wrapper.encodings['train']
    if cmd_args.model == 'hsmm':
        pred = hsmm_wrapper.predictions['train']
    if cmd_args.model == 'eval':
        eval = OddManOut(nbc_wrapper, hsmm_wrapper)
        eval.evaluate()
