#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:33:29 2022

@author: fht
"""



import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import time



def skMetrics(yLabel, yPredScore):
    # AUPR AUC
    AUPR = round(average_precision_score(yLabel, yPredScore),4)
    AUC = round(roc_auc_score(yLabel, yPredScore),4)
    # f1 = f1_score(yLabel, yPred, average='macro')
    # accuracy = accuracy_score(yLabel, yPred)

    metricLabel = ['AUPR', 'AUC', 'F1_score', 'ACC', 'Rec', 'Spec', 'Prec']
    metricResult = [AUPR, AUC, 2., 3., 4., 5., 6.]
    return metricLabel, metricResult


def calMetrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    aupr, auc = round(aupr[0, 0],4), round(auc[0, 0],4)

    metricLabel = ['AUPR', 'AUC', 'F1_score', 'ACC', 'Rec', 'Spec', 'Prec']
    metricResult = [round(temp,4) for temp in [aupr, auc, f1_score, accuracy, recall, specificity, precision]]
    return metricLabel, metricResult



def now_to_date(format_string="%Y%m%d %H%M%S"):
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


