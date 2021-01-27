import json

class Args:
    def __init__(self):
        return

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
    args.nbc_chunk_size = args_dict['nbc_chunk_size']
    args.nbc_sliding_chunk_stride = args_dict['nbc_sliding_chunk_stride']
    args.nbc_label_method = args_dict['nbc_label_method']
    args.nbc_features = args_dict['nbc_features']
    args.nbc_output_type = args_dict['nbc_output_type']
    args.nbc_preprocessing = args_dict['nbc_preprocessing']
    args.vae_hidden_size = args_dict['vae_hidden_size']
    args.vae_batch_size = args_dict['vae_batch_size']
    args.vae_beta = args_dict['vae_beta']
    return args

if __name__ == '__main__':
    args = deserialize('vae8_actions_beta=0')
    print(vars(args))
