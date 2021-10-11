import sys
sys.path.append('../miblock/')
from utils import Registry
newclass = Registry('newclass')
@newclass.register_module()
class Thisclass():
    def __init__(self,name):
        self.name=name

print(newclass.get('Thisclass'))


