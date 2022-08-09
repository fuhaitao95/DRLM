#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:32:57 2022

@author: fht
"""


import argparse
from functools import reduce
import numpy as np
import os
import pandas as pd

import random

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from tqdm import trange


from processLinkData import getData
from utils.utils import calMetrics, now_to_date, skMetrics



def crossValidation(clf, X, y, nFolds):
    random.seed(0); np.random.seed(0)
    kf = KFold(n_splits=nFolds,shuffle=True,random_state=0)
    metricLs = []; allPred = np.zeros((X.shape[0],))
    for traiIndex, testIndex in kf.split(X):
        XTra, yTra = X[traiIndex], y[traiIndex]
        XTes, yTes = X[testIndex], y[testIndex]
        classifier = clf.fit(XTra, yTra)
        yPredScore = classifier.predict_proba(XTes)[:,1]
        yPredLael = classifier.predict(XTes)
        allPred[testIndex] = yPredScore
        # metricLabel, metricSk = skMetrics(yTes, yPredScore)
        metricLabel, metricSe = calMetrics(yTes, yPredScore)
        metric = metricSe
        metricLs.append(metric)
    metricAr = np.array(metricLs).mean(axis=0)
    return metricLabel, metricAr, allPred
def conductCV(clfName, XDataLabels, XDataDict, yLabels, nFolds):
    clfDict = {'LR': LogisticRegression(random_state=0), 'RF': RandomForestClassifier(random_state=0),
               'SVM': make_pipeline(StandardScaler(), SVC(probability=True, gamma='auto')),
               'XGB': XGBClassifier(random_state=0), 'MLP': MLPClassifier(random_state=0),
               'NB': GaussianNB()}

    clf = clfDict[clfName]

    metricData = []; predDF = pd.DataFrame()
    for XLabelI in trange(len(XDataLabels)):
        XLabel = XDataLabels[XLabelI]
        XData = np.array(XDataDict[XLabel])
        metricLabel, metric, allPred = crossValidation(clf, XData, yLabels, nFolds)
        metricData.append(list(metric))
        predDF[XDataLabels[XLabelI]] = allPred
    metricDF = pd.DataFrame(metricData, columns=metricLabel, index=XDataLabels)

    return metricDF, predDF



def CVMain(args, dataObj):
    dataName = args.dataName
    nFolds = args.nFolds
    outFile = args.outFile
    clfName = args.clfName
    featureInd = args.featureInd

    outDir = outFile + '/' + clfName + '/'

    XDataLabels, XDataDict, yLabelsMul = dataObj.XDataLabels, dataObj.XDataDict, dataObj.yLabelsMul


    XDataLabels = [XDataLabels[featureInd]]

    metrics = []; predDFs = []
    iLabels = 0
    for iLabels in range(yLabelsMul.shape[1]):
        yLabels = yLabelsMul[:,iLabels]
        metricDF, predDF = conductCV(clfName, XDataLabels, XDataDict, yLabels, nFolds)
        metrics.append(metricDF)
        predDFs.append(predDF)

    metricAve = reduce(lambda x,y:x+y, metrics) / yLabelsMul.shape[1]

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    outMetrics = outDir+'_'.join([dataName, clfName, 'CV', 'result', 'metrics.csv'])
    with open(outMetrics, 'a') as fobj:
        fobj.write('\n\n'+','.join([str(key)+': '+str(value) for key,value in vars(args).items()])+'\n')
    metricAve.to_csv(outMetrics, mode='a', float_format='%.4f')

    outPred = outDir+'_'.join([dataName, clfName, 'CV', 'result', 'predictedScores_all.csv'])
    predDFs[0].to_csv(outPred, mode='a', header=True, index=False, float_format='%.6f')
    for iLabels in range(1, yLabelsMul.shape[1]):
        predDFs[iLabels].to_csv(outPred, mode='a', header=False, index=False, float_format='%.6f')
    return metricAve


def getArgs():
    parser = argparse.ArgumentParser(description='Get link data')
    parser.add_argument('--expName', default='expCV')

    parser.add_argument('--clfName', default='RF', choices=['NB', 'LR', 'RF', 'XGB'])
    parser.add_argument('--dataPrefix', default='../../dataset/')
    parser.add_argument('--dataName', default='drugDis', choices=['drugDis', 'DDI', 'drugMiRNA'])
    parser.add_argument('--embeddedName', default='EmbeddedZLast_mean.csv', type=str)
    # choices=['EmbeddedZBeforeAE_mean.csv', 'EmbeddedZBeforeCluster_mean.csv', 'EmbeddedZBeforeDNN_mean.csv', 'EmbeddedZLast_mean.csv']
    parser.add_argument('--featureInd', default=-1, type=int) # DDA DDI,01234; DSE,0123456; DMA,0123

    parser.add_argument('--nFolds', default=5, type=int, choices=[2, 5, 10])
    parser.add_argument('--outFile', default='output/result', type=str)
    
    parser.add_argument('--dimPara', default=[128, 32], type=int, nargs='+')
    
    args = parser.parse_args()
    return args


def CVOperator():
    args = getArgs()

    dataPrefix = args.dataPrefix
    dataName = args.dataName
    embeddedName = args.embeddedName

    dataObj = getData(args, dataPrefix, dataName, embeddedName)
    resultCV = CVMain(args, dataObj)
    return resultCV


if __name__ == '__main__':
    resultCV = CVOperator()


