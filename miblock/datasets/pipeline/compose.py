from ..builder import PIPELINE
@PIPELINE.register_module()
class Compose(object):
    def __init__(self,transform):
        self.tansform = build(transform)

    def __call__(self,data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


