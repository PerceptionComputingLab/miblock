import sys
sys.path.append('../../')
from utils import PIPELINE
@PIPELINE.register_module()
class Crop():
    def __init__(self):
        pass
    def __call__(self,data):


        return data

