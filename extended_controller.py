import numpy as np
import pandas as pd
from autoencoder_wrapper import AutoencoderWrapper, AutoencoderConcatWrapper
from hsmm_wrapper import HSMMWrapper
import os
import sys

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
sys.path.append(os.environ['NBC_ROOT'])
from nbc import NBC, obj_names
from nbc_wrapper import NBCWrapper
import config
import controller

def initialize(args):
    if args.nbc_features == 'special_obj_concat':
        configs = []
        for obj in obj_names:
            configs.append('autoencoder_{}'.format(obj))
        controller.nbc_wrapper = NBCWrapper(args)
        controller.autoencoder_wrapper = AutoencoderConcatWrapper(configs)
        controller.hsmm_wrapper = HSMMWrapper(args, steps=controller.nbc_wrapper.nbc.steps, autoencoder_wrapper=controller.autoencoder_wrapper, device='cuda')

if __name__ == '__main__':
    configs = []
    for obj in obj_names:
        configs.append('autoencoder_{}'.format(obj))

    args = config.deserialize('hsmm')
    args.nbc_features = 'special_obj_concat'
    steps = NBC(args).steps

    autoencoder_wrapper = AutoencoderConcatWrapper(configs)
    hsmm_wrapper = HSMMWrapper(args, steps=steps, autoencoder_wrapper=autoencoder_wrapper, device='cuda')
