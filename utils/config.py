import numpy as np
import os


class SaveConfig(object):
    """
    Helper to save configuration files.

    This wraps np.savez() into a nice class.
    """

    def __init__(self, file_name, config_name='config'):
        if os.path.exists(file_name):
            print('Warning: Config file {} will be overwritten'.format(
                file_name))
        self.file_name = file_name
        self.config_name = config_name

    def save(self, **kwds):
        np.savez(self.file_name, **kwds)
        print('Saved {} {} to {}'.format(
            self.config_name, kwds, self.file_name))


class LoadConfig(object):
    """
    Helper to load configuration items.

    This wraps np.load() into a nice class and returns
    a dictionary of configuration values.
    """

    def __init__(self, file_name, config_name='config'):
        if not os.path.exists(file_name):
            raise Exception('Config file {} not found'.format(file_name))
        self.file_name = file_name
        self.config_name = config_name

    def load(self):
        config = {}
        with np.load(self.file_name) as data:
            for d in data.files:
                config[d] = data[d]

        print('Loaded {} {} from {}'.format(
            self.config_name, config, self.file_name))

        return config
