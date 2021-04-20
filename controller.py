from hsmm_wrapper import HSMMWrapper
from input_modules import InputModule, ConvertToChunks, ConvertToSessions
from input_modules import ReducePCA
import os
import pandas as pd

HSMM_WRAPPER = None
INPUT_MODULE = None


def initialize(fpath):
    global INPUT_MODULE
    global HSMM_WRAPPER
    if 'hsmm_' in fpath:
        # hsmm_wrapper = VirtualHSMMWrapper(fpath)
        HSMM_WRAPPER = HSMMWrapper(fpath, device='cuda')
        # merge_similar_states(hsmm_wrapper)
        INPUT_MODULE = HSMM_WRAPPER.input_module
    else:
        fname = os.path.basename(fpath).replace('.json', '')
        INPUT_MODULE = InputModule.load_from_config(fname)


def get_input_data(session, type='dev'):
    global INPUT_MODULE
    z = INPUT_MODULE.z[type]
    steps = INPUT_MODULE.steps[type]
    lengths = INPUT_MODULE.lengths[type]
    for i, key in enumerate(steps.keys()):
        if key[0] == session:
            return (z[i][:lengths[i]], steps[key])
    return None


def get_hsmm_input_encodings(session, type='dev'):
    global HSMM_WRAPPER
    predictions = HSMM_WRAPPER.predictions[type]
    chunks = ConvertToChunks(HSMM_WRAPPER.input_module)
    module = ConvertToSessions(ReducePCA(chunks), 2)
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
                except IndexError:
                    data.append({
                        'start_step': int(steps[key][j]),
                        'end_step': int(steps[key][j] + 9),
                        'encoding': z[i][j],
                        'label': int(predictions[i][j]),
                        'timestamp': get_timestamp(session, steps[key][j])
                    })
    return pd.DataFrame(data)


def get_predictions(session, type='dev'):
    global HSMM_WRAPPER
    steps = HSMM_WRAPPER.input_module.steps[type]
    for i, key in enumerate(steps.keys()):
        if key[0] == session:
            return HSMM_WRAPPER.predictions[type][i], steps[key]
    assert len(predictions) == 1
    return predictions[0]


def get_timestamp(session, step, type='dev'):
    global INPUT_MODULE
    steps = INPUT_MODULE.steps[type]
    start_step = -1
    for key in steps.keys():
        if key[0] == session:
            try:
                start_step = int(steps[key][0][0])
            except IndexError:
                start_step = int(steps[key][0])
            break
    assert not start_step == -1
    return float((step - start_step) / 90.)


if __name__ == '__main__':
    initialize('hsmm_max_objs_indices')
    # encodings = get_encodings('17_1c_task1', 'dev')
    predictions, steps = get_predictions('17_1c_task2', 'dev')
    print(predictions, steps)
