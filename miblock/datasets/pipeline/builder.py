class Registry():
    def __init__(self,name):
        self._name = name
        self._module_dict = dict()

    def get(self,key):
        #print(self._module_dict)
        return self._module_dict[key]

    def register_module(self, name=None, force=False, module=None):
        def _register(cls):
            #print('cls ', cls)
            if name == None:
                self._module_dict[cls.__name__] = cls
                return cls
            elif isinstance(name, str):
                self._module_dict[name] = cls
            else:
                raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        return _register
DATASETS = Registry('dataset')
PIPELINE = Registry('pipeline')


def build(key,register):
    args = key.copy()
    if isinstance(key,dict):
        t = register.get(args.pop('type'))
        return t(**args)

