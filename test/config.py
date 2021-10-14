cfg = dict(
    img_dir = 'W:\LITS17\image',
    lab_dir = 'W:\LITS17\label',
    train_pipeline = [
        dict(type='LoadImage'),
        dict(type='LoadLabel'),
        ]
    )
import json 
cfg2=json.dumps(cfg)
print(cfg2)