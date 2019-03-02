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

PARAMS = {
    'PATHS': PATHS,
    'SEED': 42,
    'NB_FOLDS': 4,
    'SIDE': 384,
    'INVERSE': 3,
    'BATCH_SIZE': 10,
    'NB_EPOCHS': 31,
    'LR': 1e-5,
    'EXP_GAMMA': .8,
    'CUDA_DEVICES': [0, 1],
}

MEAN = [0.15817458]
STD = [0.16162706]

PARAMS = easydict.EasyDict(PARAMS)
