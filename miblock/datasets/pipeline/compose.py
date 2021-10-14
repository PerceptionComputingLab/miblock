import sys
from .builder import PIPELINE,build
@PIPELINE.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """
    def __init__(self,transforms):
        self.transform = []
        for transform in transforms:
            self.transform.append(build(transform,PIPELINE))


    def __call__(self,data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transform:
            print(t)
            data = t(data)
            if data is None:
                return None
        return data



