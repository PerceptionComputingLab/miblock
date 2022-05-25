import json
import os


class Config:
    """
    Load configuration file
    """

    def load(path):
        """If the path is file,then turn the configuration file into dictionary,
           if the path is folder,then turn files into dictionaries in turn and add them into a list
        Args:
            path(file name or directory):Path of configuration file or folder
        """
        if os.path.isfile(path):
            with open(path, 'r') as f:
                cfg = json.load(f)
                return cfg
        elif os.path.isdir(path):
            allpath = os.listdir(path)
            cfg = []
            for filepath in allpath:
                filepath = os.path.join('%s%s' % (path, filepath))
                with open(path, 'r') as f:
                    cfg.append(json.load(f))
            return cfg
        else:
            print("Please give the correct path")
