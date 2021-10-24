class Registry():
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self,name):
        self._name = name
        self._module_dict = dict()

    def get(self,key):
        """Return a class by the name of class
        Args:
            key(name):name of the class
        """
        return self._module_dict[key]

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        def _register(cls):
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
MODELS = Registry("model")

def build(cfg,register):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        register (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed object.
    """
    args = cfg.copy()
    if isinstance(cfg,dict):
        t = register.get(args.pop('type'))
        return t(**args)



