class Config:
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.default_values = {}

    def get_property(self, name):
        if name not in self.cfg:
            if name in self.default_values:
                return self.default_values[name]
            else:
                return None
        return self.cfg[name]


class NBCConfig(Config):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.default_values = {
            'nbc_subsample': 9,
            'nbc_dynamic_only': True,
            'nbc_train_sequencing': 'session',
            'nbc_dev_sequencing': 'session',
            'nbc_test_sequencing': 'session',
            'nbc_chunk_size': 10,
            'nbc_sliding_chunk_stride': 3,
            'nbc_label_method': 'none',
            'nbc_features': [],
            'nbc_output_type': 'classifier',
            'nbc_preprocessing': []
        }

    @property
    def nbc_subsample(self):
        return self.get_property('nbc_subsample')

    @property
    def nbc_dynamic_only(self):
        return self.get_property('nbc_dynamic_only')

    @property
    def nbc_train_sequencing(self):
        return self.get_property('nbc_train_sequencing')

    @property
    def nbc_dev_sequencing(self):
        return self.get_property('nbc_dev_sequencing')

    @property
    def nbc_test_sequencing(self):
        return self.get_property('nbc_test_sequencing')

    @property
    def nbc_chunk_size(self):
        return self.get_property('nbc_chunk_size')

    @property
    def nbc_sliding_chunk_stride(self):
        return self.get_property('nbc_sliding_chunk_stride')

    @property
    def nbc_label_method(self):
        return self.get_property('nbc_label_method')

    @property
    def nbc_features(self):
        return self.get_property('nbc_features')

    @property
    def nbc_output_type(self):
        return self.get_property('nbc_output_type')

    @property
    def nbc_preprocessing(self):
        return self.get_property('nbc_preprocessing')
