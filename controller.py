import numpy as np
from autoencoder_wrapper import AutoencoderWrapper
from hsmm_wrapper import HSMMWrapper
from sklearn import preprocessing
import os
import sys
import argparse

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
sys.path.append(os.environ['NBC_ROOT'])
from nbc_wrapper import NBCWrapper
import config

nbc_wrapper = None
autoencoder_wrapper = None
hsmm_wrapper = None

def initialize(args, model):
    global nbc_wrapper
    global autoencoder_wrapper
    global hsmm_wrapper
    if model in ['nbc', 'autoencoder', 'hsmm']:
        nbc_wrapper = NBCWrapper(args)
    if model in ['autoencoder', 'hsmm']:
        autoencoder_wrapper = AutoencoderWrapper(args, nbc_wrapper)
    if model in ['hsmm']:
        hsmm_wrapper = HSMMWrapper(args, nbc_wrapper, autoencoder_wrapper)

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

def get_predictions(args, type='train'):
    global hsmm_wrapper
    assert hsmm_wrapper is not None
    sessions = hsmm_wrapper.sequences[type][0]
    pred = hsmm_wrapper.predictions[type]
    return sessions, pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='hidden=8_beta=10')
    parser.add_argument('--model', type=str, choices=['nbc', 'autoencoder', 'hsmm'], default='autoencoder')
    cmd_args = parser.parse_args()
    args = config.deserialize(cmd_args.config)
    initialize(args, cmd_args.model)
    if cmd_args.model == 'nbc':
        print(nbc_wrapper.nbc.steps['train'].items())
    if cmd_args.model == 'autoencoder':
        z = autoencoder_wrapper.encodings['train']
    if cmd_args.model == 'hsmm':
        pred = hsmm_wrapper.predictions['train']
