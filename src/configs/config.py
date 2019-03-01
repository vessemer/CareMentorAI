import os
import easydict


DATA_ROOT = '../data'

PATHS = {
    'IMAGES': 'images',
    'CSV': 'descr',
}

for k, v in PATHS.items():
    PATHS[k] = os.path.join(DATA_ROOT, v)

PATHS['DATA'] = DATA_ROOT
PATHS = easydict.EasyDict(PATHS)
