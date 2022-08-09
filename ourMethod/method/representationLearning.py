#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:08:40 2022

@author: fht
"""


import argparse
import sys
import pkg_resources
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from desc_modified.models_modified.desc import getdims
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import pickle
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
from desc_modified.models_modified import network


import os
from sklearn.preprocessing import StandardScaler
from preprocess import processLINCS


def getRawData(dataPrefix):
    trainName = 'output/processedData/rawExpressionData.pkl'
    train_data = pickle.load(open(trainName, 'rb'))

    labelInitName = 'output/processedData/drugLabelInit.pkl'
    label_init = pickle.load(open(labelInitName, 'rb'))

    labelOheName = 'output/processedData/drugLabelOneHotFile.pkl'
    label_ohe = pickle.load(open(labelOheName, 'rb'))

    return train_data, label_init, label_ohe


def normal(x):
    preprocessor = StandardScaler().fit(x)
    x_norm = preprocessor.transform(x)
    return x_norm


def getRep(train_data, label_init, label_ohe, args, dimPara):
    dataPrefix = args.dataPrefix
    use_ae_weights = args.use_ae_weights
    pretrain_epochs = args.pretrain_epochs

    print(train_data.shape)
    # % get dims
    # getdims(shape)
    adata = normal(train_data.T.values)

    tempPrefix = 'output/' + \
        '_'.join([str(temp) for temp in dimPara]) + '_' + 'result_model/'

    dsn_run = network.DescModel(dims=[adata.shape[1]]+dimPara, x=adata, tol=0.05,
                                batch_size=256, louvain_resolution=[0.8], n_clusters=18, use_ae_weights=use_ae_weights,
                                save_dir=tempPrefix, label_init=label_init, label_ohe=label_ohe, pretrain_epochs=pretrain_epochs)
    if (not use_ae_weights) or dsn_run.denovo:
        dsn_run.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        Embedded_z = dsn_run.fit(maxiter=300)

    label_contain = {}
    label_contain['EmbeddedZLast'] = dsn_run.Embedded_z
    label_contain['EmbeddedZBeforeAE'] = dsn_run.featuresBeforeAE
    label_contain['EmbeddedZBeforeCluster'] = dsn_run.featuresBeforeCluster
    label_contain['EmbeddedZBeforeDNN'] = dsn_run.featuresBeforeDNN

    labelContainedName = tempPrefix + args.labelContainedName
    with open(labelContainedName, 'wb') as f:
        pickle.dump(label_contain, f)

    cellDrugInfoM = pd.read_csv('output/processedData/drugCellSampleInfo.csv')

    for key, value in label_contain.items():
        embeddings = pd.DataFrame(value)
        embeddings.index = train_data.columns
        embeddings.insert(0, 'iname', cellDrugInfoM['drugName'].values)
        embeddedZName = tempPrefix+key+'.csv'
        embeddings.to_csv(embeddedZName)

        z_mean = embeddings.groupby(['iname']).mean()
        embeddedZMeanName = tempPrefix+key+'_mean.csv'
        z_mean.to_csv(embeddedZMeanName)
    return z_mean, label_contain


def getArgs():
    parser = argparse.ArgumentParser(description='Get link data')
    parser.add_argument(
        '--expName', default='representationLearning', type=str)

    parser.add_argument(
        '--dataPrefix', default='../../dataset/mainData/', type=str)
    parser.add_argument('--labelContainedName',
                        default='label_contain.pkl', type=str)
    parser.add_argument('--dimPara', default=[128, 32], type=int, nargs='+')

    parser.add_argument('--pretrain_epochs', default=30, type=int)
    parser.add_argument('--use_ae_weights', default=0,
                        type=int, choices=[0, 1])

    args = parser.parse_args()
    return args


def representOperator():
    args = getArgs()
    # print(args)
    # return args,args

    dataPrefix = args.dataPrefix
    dimPara = args.dimPara

    train_data, label_init, label_ohe = getRawData(dataPrefix)

    z_mean, label_contain = getRep(
        train_data, label_init, label_ohe, args, dimPara)

    return z_mean, label_contain


if __name__ == '__main__':
    z_mean, label_contain = representOperator()
