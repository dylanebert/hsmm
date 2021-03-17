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

hsmm_wrapper = None

def initialize(args):
    global hsmm_wrapper
    hsmm_wrapper = HSMMWrapper(args)

def get_encodings(args, session, type='train'):
    global hsmm_wrapper
    assert hsmm_wrapper is not None
    autoencoder_wrapper = hsmm_wrapper.autoencoder_wrapper
    nbc_wrapper = autoencoder_wrapper.nbc_wrapper
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
    global hsmm_wrapper
    assert hsmm_wrapper is not None
    autoencoder_wrapper = hsmm_wrapper.autoencoder_wrapper
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
    global hsmm_wrapper
    assert hsmm_wrapper is not None
    autoencoder_wrapper = hsmm_wrapper.autoencoder_wrapper
    nbc_wrapper = autoencoder_wrapper.nbc_wrapper
    keys = list(nbc_wrapper.nbc.steps[type].keys())
    sessions = np.unique([key[0] for key in keys]).tolist()
    return sessions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['nbc', 'autoencoder', 'hsmm', 'eval'], default='autoencoder')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    cmd_args = parser.parse_args()
    args = config.deserialize(cmd_args.config)
    if cmd_args.model == 'nbc':
        nbc_wrapper = NBCWrapper(args)
    elif cmd_args.model == 'autoencoder':
        autoencoder_wrapper = AutoencoderWrapper(args)
    elif cmd_args.model == 'hsmm':
        hsmm_wrapper = HSMMWrapper(args, device=cmd_args.device)
