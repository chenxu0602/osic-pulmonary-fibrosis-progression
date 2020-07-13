import os, sys
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from functools import partial

import numpy as np
import pandas as pd
import random
import math

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error

from PIL import Image
import cv2
import pydicom

import lightgbm as lgb
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

from pathlib import Path
import shutil
import tempfile

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots


ID = 'Patient_Week'
TARGET = 'FVC'
SEED = 42
N_FOLD = 4

train = pd.read_csv("train.csv")
train[ID] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)
print(train.shape)


output = pd.DataFrame()
gb = train.groupby('Patient')
tk0 = tqdm(gb, total=len(gb))

for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby('Weeks'):
        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}
        tmp = tmp.drop(columns='Patient_Week').rename(columns=rename_cols)
        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']
        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')
        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])
    
train = output[output['Week_passed']!=0].reset_index(drop=True)
print(train.shape)

submission = pd.read_csv('sample_submission.csv')
print(submission.shape)

"""
folds = train[[ID, 'Patient', TARGET]].copy()
#Fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds['Patient'].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
folds.head()
"""


logdir = Path("./logs") / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# features
cat_features = ['Sex', 'SmokingStatus']
num_features = [c for c in train.columns if (train.dtypes[c] != 'object') & (c not in cat_features)]
features = num_features + cat_features
drop_features = [ID, TARGET, 'predict_Week', 'base_Week']
features = [c for c in features if c not in drop_features]


base_fvc = tf.feature_column.numeric_column("base_FVC")
base_pct = tf.feature_column.numeric_column("base_Percent")
base_age = tf.feature_column.numeric_column("base_Age")
base_wkp = tf.feature_column.numeric_column("Week_passed")

sex = tf.feature_column.categorical_column_with_vocabulary_list("Sex", ["Male", "Female"])
smoke = tf.feature_column.categorical_column_with_vocabulary_list(
    "SmokingStatus", ["Ex-smoker", "Never Smoked", "Currently smokes"])

sex_1hot = tf.feature_column.indicator_column(sex)
smoke_1hot = tf.feature_column.indicator_column(smoke)

feature_columns = [
    base_fvc,
    base_pct,
    base_age,
    base_wkp,
    sex_1hot,
    smoke_1hot,
]

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("FVC")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.repeat().batch(batch_size)
    return ds

batch_size = 5

from sklearn.model_selection import train_test_split
train, test = train_test_split(train, test_size=0.2)
train, valid = train_test_split(train, test_size=0.2)

train_ds = df_to_dataset(train, batch_size=batch_size)
valid_ds = df_to_dataset(valid, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.models.Sequential([
    feature_layer,
    tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

def get_callbacks(name):
      return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

model.compile(loss="mse", optimizer="adam", metrics=["mae"])
history = model.fit(train_ds, 
          steps_per_epoch=len(train) // batch_size, 
          validation_steps=len(valid) // batch_size,
          epochs=500, validation_data=valid_ds, 
          verbose=2,
          callbacks=get_callbacks("osic"))

#loss, mae = model.evaluate(test_ds)
#print(f"loss = {loss}, mae = {mae}")