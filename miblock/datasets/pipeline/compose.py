import sys
from .builder import PIPELINE,build
@PIPELINE.register_module()
class Compose(object):
    def __init__(self,transforms):
        self.transform = []
        for transform in transforms:
            self.transform.append(build(transform,PIPELINE))

    def __call__(self,data):
        for t in self.transform:
            #print(t)
            data = t(data)
            if data is None:
                return None
        return data



