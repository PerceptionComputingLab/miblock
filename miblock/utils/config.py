import json
class Config:
    def load(filename):
        with open(filename,'r') as f:
            cfg = json.load(f)
            
            return cfg



