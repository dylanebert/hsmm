import json
import argparse
from autoencoder import AutoencoderWrapper

class Args:
    def __init__(self):
        self.nbc_subsample = 9
        self.nbc_dynamic_only = True
        self.nbc_train_sequencing = 'actions'
        self.nbc_dev_sequencing = 'actions'
        self.nbc_test_sequencing = 'actions'
        self.nbc_label_method = 'hand_motion_rhand'
        self.nbc_features = ['velY:RightHand', 'relVelZ:RightHand']

        self.nbc_output_type = 'classifier'
        self.nbc_preprocessing = ['robust', 'min-max']

        self.vae_hidden_size = 8
        self.vae_batch_size = 10
        self.vae_beta = 10

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
    args.nbc_label_method = args_dict['nbc_label_method']
    args.nbc_features = args_dict['nbc_features']
    args.nbc_output_type = args_dict['nbc_output_type']
    args.nbc_preprocessing = args_dict['nbc_preprocessing']
    args.vae_hidden_size = args_dict['vae_hidden_size']
    args.vae_batch_size = args_dict['vae_batch_size']
    args.vae_beta = args_dict['vae_beta']
    return args

def get_encodings(args, type='train'):
    autoencoder_wrapper = AutoencoderWrapper(args)
    z, y = autoencoder_wrapper.get_encodings(type=type)
    return z, y

if __name__ == '__main__':
    args = Args()
    serialize(args, 'test')
