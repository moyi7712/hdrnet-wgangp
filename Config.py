import yaml


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

    def HyperPara(self):
        return dict_to_object(self._config['HyperPara'])

    def Dataset(self):
        return dict_to_object(self._config['Dataset'])