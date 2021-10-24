import sys
sys.path.append('../../')
from utils import PIPELINE,build
@PIPELINE.register_module()
class Compose(object):
    """Compose  transforms together.

    Args:
        transforms: A list of transform class dict.
    """
    def __init__(self,transforms):
        self.transform = []
        for transform in transforms:
            self.transform.append(build(transform,PIPELINE))


    def __call__(self,data):
        """Make tansforms run sequentially.

        Args:
            data:Filenames of the data or dict that contain the informations of the data

        Returns:
           dict: Transformed data.
        """

        for t in self.transform:
            print(t)
            data = t(data)
            if data is None:
                return None
        return data



