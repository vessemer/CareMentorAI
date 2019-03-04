import os
import easydict


DATA_ROOT = '../data'

PATHS = {
    'IMAGES': 'images',
    'CSV': 'descr',
    'SUB': 'subs',
    'MODELS': 'models',
    'LOGDIR': 'logdir',
}

for k, v in PATHS.items():
    PATHS[k] = os.path.join(DATA_ROOT, v)

PATHS['DATA'] = DATA_ROOT
PATHS = easydict.EasyDict(PATHS)

PARAMS = {
    'PATHS': PATHS,
    'SEED': 42,
    'NB_FOLDS': 5,
    'SIDE': 384,
    'INVERSE': 3,
    'BATCH_SIZE': 16,
    'THRESHOLD': .5,
    'NB_EPOCHS': 255,
    'LR': 1e-5,
    'EXP_GAMMA': .99,
    'CUDA_DEVICES': [0, 1],
}

MEAN = [0.15817458]
STD = [0.16162706]

PARAMS = easydict.EasyDict(PARAMS)
