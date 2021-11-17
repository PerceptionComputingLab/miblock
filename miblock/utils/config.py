import json
import os
class Config:
    def load(filepath):
        if os.path.isfile(filepath):
            with open(filepath,'r') as f:
                cfg = json.load(f)          
                return cfg
        elif os.path.isdir(filepath):
            allpath = os.listdir(filepath)
            cfg = []
            for path in allpath:
                path = os.path.join('%s%s' % (filepath, path))
                with open(path,'r') as f:
                    cfg.append(json.load(f))          
            return cfg
        else:
            print("Please give the correct filepath")



