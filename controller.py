import numpy as np
import pandas as pd
import os
import sys
import argparse
import input_modules
from input_modules import InputModule, ConvertToChunks, ConvertToSessions, ReducePCA
from hsmm_postprocessing import merge_similar_states

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT'
sys.path.append(os.environ['NBC_ROOT'])
import config
from hsmm_wrapper import HSMMWrapper, VirtualHSMMWrapper

hsmm_wrapper = None
input_module = None

def initialize(fpath):
    global input_module
    global hsmm_wrapper
    if 'hsmm_' in fpath:
        #hsmm_wrapper = VirtualHSMMWrapper(fpath)
        hsmm_wrapper = HSMMWrapper(fpath, device='cuda')
        merge_similar_states(hsmm_wrapper)
        input_module = hsmm_wrapper.input_module
    else:
        fname = os.path.basename(fpath).replace('.json', '')
        input_module = InputModule.load_from_config(fname)

def get_input_data(session, type='dev'):
    global input_module
    z = input_module.z[type]
    steps = input_module.steps[type]
    lengths = input_module.lengths[type]
    for i, key in enumerate(steps.keys()):
        if key[0] == session:
            return (z[i][:lengths[i]], steps[key])
    return None

def get_hsmm_input_encodings(session, type='dev'):
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
                try:
                    data.append({
                        'start_step': int(steps[key][j][0]),
                        'end_step': int(steps[key][j][-1]),
                        'encoding': z[i][j],
                        'label': int(predictions[i][j]),
                        'timestamp': get_timestamp(session, steps[key][j][0])
                    })
                except:
                    data.append({
                        'start_step': int(steps[key][j]),
                        'end_step': int(steps[key][j] + 9),
                        'encoding': z[i][j],
                        'label': int(predictions[i][j]),
                        'timestamp': get_timestamp(session, steps[key][j])
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
    global input_module
    steps = input_module.steps[type]
    start_step = -1
    for key in steps.keys():
        if key[0] == session:
            try:
                start_step = int(steps[key][0][0])
            except:
                start_step = int(steps[key][0])
            break
    assert not start_step == -1
    return float((step - start_step) / 90.)

if __name__ == '__main__':
    initialize('hsmm_max_objs_indices')
    #encodings = get_encodings('17_1c_task1', 'dev')
    predictions, steps = get_predictions('17_1c_task2', 'dev')
    print(predictions, steps)
