# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:12:56 2021

@author: alank
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


print(tf.version)
string = tf.Variable("hi", str)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


floating = tf.Variable(2.345,tf.float64)
print(floating)
print(string)



rank2_tensor = tf.Variable([["string","is"],["Very","long"]], str)
print(rank2_tensor)
tf.rank(rank2_tensor)
rank2_tensor.shape
string.shape


tensor1 = tf.ones([1,2,3])
print(tensor1)

tensor2 = tf.reshape(tensor1,[2,3,1])
tensor2

tenor3 = tf.reshape(tensor2,[3,-1])
tenor3

matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

matrix_tensor = tf.Variable(matrix, tf.int32)

print(matrix_tensor.shape)
print(tf.rank(matrix_tensor))

matrix_tensor[0,2]
matrix_tensor[:,0]
matrix_tensor[1:2,0]


################################################ Lin regression #############################

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.sex.value_counts().plot(kind='barh')
dftrain.age.hist(bins =20)
dftrain['class'].value_counts().plot(kind='barh')


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


feature_columns =[]
for feature in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))

for feature in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature,dtype = tf.float32))
    
print(feature_columns)


def make_input_fn(data_df, label_df, shuffle=True, n_epochs=10, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(n_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, False,1)


linear_est = tf.estimator.LinearClassifier(feature_columns= feature_columns)
linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn)
print(result['accuracy'])


pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins = 20, title='Predicted Probabilities')



