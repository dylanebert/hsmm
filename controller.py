import numpy as np
import pandas as pd
import os
import sys
import argparse
import input_modules
from input_modules import ConvertToChunks, ConvertToSessions, ReducePCA

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
sys.path.append(os.environ['NBC_ROOT'])
import config
from hsmm_wrapper import HSMMWrapper

hsmm_wrapper = None

def initialize(fpath):
    global hsmm_wrapper
    hsmm_wrapper = HSMMWrapper(fpath, device='cuda')

def get_encodings(session, type='dev'):
    global hsmm_wrapper
    predictions = hsmm_wrapper.predictions[type]
    module = ConvertToSessions(ReducePCA(ConvertToChunks(hsmm_wrapper.input_module)))
    z = module.z[type]
    steps = module.steps[type]
    lengths = module.lengths[type]
    data = []
    for i, key in enumerate(steps.keys()):
        if key[0] == session:
            for j in range(int(lengths[i])):
                print(steps[key].shape)
                data.append({
                    'start_step': int(steps[key][j][0]),
                    'end_step': int(steps[key][j][-1]),
                    'encoding': z[i][j],
                    'label': int(predictions[i][j]),
                    'timestamp': get_timestamp(session, steps[key][j][0])
                })
    return pd.DataFrame(data)

def get_predictions(session, type='dev'):
    global hsmm_wrapper
    steps = hsmm_wrapper.input_module.steps[type]
    for i, key in enumerate(steps.keys()):
        if key[0] == session:
            return hsmm_wrapper.predictions[type][i], steps[key]
    assert len(predictions) == 1
    return predictions[0]

def get_timestamp(session, step, type='dev'):
    global hsmm_wrapper
    steps = hsmm_wrapper.input_module.steps[type]
    start_step = -1
    for key in steps.keys():
        if key[0] == session:
            start_step = int(steps[key][0][0])
            break
    assert not start_step == -1
    return float((step - start_step) / 90.)

if __name__ == '__main__':
    initialize('hsmm_max_objs_indices')
    #encodings = get_encodings('17_1c_task1', 'dev')
    predictions, steps = get_predictions('17_1c_task2', 'dev')
    print(predictions, steps)
