from tqdm import tqdm
import xml.dom.minidom

import pandas as pd
import os 
from glob import glob

import numpy as np
import cv2

import PIL
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

from ..utils.xml_parser import parse_xml
from ..configs import config


paths = glob(os.path.join(config.PATHS.CSV, 'export_gr-mri-spine*.csv'))
print(os.path.abspath(config.PATHS.CSV))

dfs = list()
for path in paths:
    df = pd.read_csv(path)
    dfs.append(df)
dfs = pd.concat(dfs)
dfs['Файлы'] = dfs['Файлы'].apply(lambda x: x.split('/n')[0])

columns_annot = [
    'Файлы разметки',
    'Шейный межпозвоночный диск - здоровый',
    'Шейный межпозвоночный диск - с подозрением на патологию',
    'Шейный межпозвоночный диск - патологический',
    'грудной межпозвоночный диск - здоровый',
    'грудной межпозвоночный диск - с подозрением на паталогию',
    'грудной межпозвоночный диск - патологический',
    'поясничный межпозвоночный диск - здоровый',
    'поясничный межпозвоночный диск - с подозрением на патологию',
    'поясничный межпозвоночный диск - патологический',
    'крестцовый межпозвоночный диск - здоровый',
    'крестцовый межпозвоночный диск - с подозрением на патологию',
    'крестцовый межпозвоночный диск - патологический'
]

dfs = dfs.drop(['Исследователь', 'Протокол'] + columns_annot, axis=1)

datas = list()
for i, row in tqdm(dfs.iterrows()):
    if pd.isna(row.XML):
        continue
    visual = row['На срезе визуализируются межпозвоночные диски']
    if pd.isna(visual) or 'Не визуализируются' in visual:
        continue
    data = parse_xml(row)
    datas.append(data)
datas = pd.concat(datas, sort=False, ignore_index=True)

datas['is_normal'] = datas.name.apply(lambda x: 'disk-zdorovyj' in x)
datas['type'] = datas.name.apply(lambda x: 'chest' if 'grudnoj' in x else 'neck')

inuse_columns = ['coords', 'name', 'id', 'filename', 'is_normal', 'type']
path = os.path.join(config.PATHS.CSV, 'labels.csv')
datas[inuse_columns].to_csv(path, index=False)
print(pd.read_csv(path).head())
