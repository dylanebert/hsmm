class Config:
    def __init__(self):
        self.cfg = {}

    def get_property(self, name):
        if name not in self.cfg:
            return None
        return self.cfg[name]


class NBCConfig(Config):

    @property
    def subsample(self):
        return self.get_property('subsample')
