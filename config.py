import json
import os
import uuid

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT envvar'
NBC_ROOT = os.environ['NBC_ROOT']

serialization_keys = {
    'nbc': ['nbc_subsample', 'nbc_dynamic_only', 'nbc_train_sequencing', 'nbc_dev_sequencing', 'nbc_test_sequencing', 'nbc_chunk_size', 'nbc_sliding_chunk_stride', \
        'nbc_features', 'nbc_label_method'],
    'autoencoder': ['vae_hidden_size', 'vae_batch_size', 'vae_beta', 'nbc_output_type', 'nbc_preprocessing'],
    'hsmm': ['sm_allow_self_transitions', 'sm_lr', 'max_k', 'overrides', 'n_classes']
}

class Args:
    def __init__(self):
        return

def args_to_id(args, model):
    args_dict = {}
    models = []
    if model in ['nbc', 'autoencoder', 'hsmm']:
        models.append('nbc')
    if model in ['autoencoder', 'hsmm']:
        models.append('autoencoder')
    if model in ['hsmm']:
        models.append('hsmm')
    for model in models:
        for key in serialization_keys[model]:
            assert key in vars(args), '{} missing from args'.format(key)
            args_dict[key] = vars(args)[key]
    return json.dumps(args_dict)

def try_find_saved(args, model):
    id = args_to_id(args)
    if model == 'nbc':
        keypath = NBC_ROOT + 'tmp/nbc/keys.json'
    elif model == 'autoencoder':
        keypath = NBC_ROOT + 'tmp/autoencoder/keys.json'
    else:
        assert model == 'hsmm'
        keypath = NBC_ROOT + 'tmp/hsmm/keys.json'
    if os.path.exists(keypath);
        with open(keypath) as f:
            keys = json.load(f)
        if id in keys:
            return keys[id]
    return None

def generate_savepath(args, model):
    id = args_to_id(args)
    if model == 'nbc':
        keypath = NBC_ROOT + 'tmp/nbc/keys.json'
    elif model == 'autoencoder':
        keypath = NBC_ROOT + 'tmp/autoencoder/keys.json'
    else:
        assert model == 'hsmm'
        keypath = NBC_ROOT + 'tmp/hsmm/keys.json'
    if os.path.exists(keypath);
        with open(keypath) as f:
            keys = json.load(f)
    else:
        keys = {}
    fid = str(uuid.uuid1())
    keys[id] = fid
    with open(keypath, 'w+') as f:
        json.dump(keys, f)
    return fid

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
    for k, v in args_dict.items():
        setattr(args, k, v)
    return args

if __name__ == '__main__':
    args = deserialize('vae8_actions_beta=0')
    print(vars(args))
