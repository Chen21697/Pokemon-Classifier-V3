#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:47:05 2020

@author: yuwenchen
"""


import numpy as np
import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('/Users/yuwenchen/Pokemon-Classifier-V3/Pokemon.csv')
data = pd.DataFrame(data)
data.head()

df = data[['Type 1', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

# extract all grass, fire and water type pokemon 
grassType = df.loc[df['Type 1'] == 'Grass']
fireType = df.loc[df['Type 1'] == 'Fire']
waterType = df.loc[df['Type 1'] == 'Water']

# let C1(grass) = [1,0,0], C2(fire) = [0,1,0], C3(water) = [0,0,1]
grassType['Label'] = '100'
fireType['Label'] = '010'
waterType['Label'] = '001'

allType = pd.concat([grassType, fireType, waterType], ignore_index=True, sort=False)
allType = shuffle(allType)

x_train = allType[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values
y_train = allType[['Label']].values.tolist()
temp = []
for i in range(len(y_train)):
    temp.append([ int(x) for x in list(y_train[i][0])])
y_train = temp
y_train = np.array(y_train)
