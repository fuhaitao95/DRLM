#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:26:37 2022

@author: fht
"""



import argparse
import copy
import numpy as np
import os
import pandas as pd
import scipy.io as scio
import scipy.sparse as sp
import sys



class linkData(object):
    def __init__(self, args, dataPrefix, dataName, embeddedName):
        self.dataPrefix = dataPrefix
        self.dataName = dataName
        self.XDataLabels = []
        self.XDataDict = dict()
        self.yLabelsMul = []
        
        self.args = args
        dimPara = args.dimPara
        tempPrefix = 'output/' + '_'.join([str(temp) for temp in dimPara]) + '_' + 'result_model/'
        
        zMeanName = os.path.join(tempPrefix, embeddedName)
        self.zMeanData = pd.read_csv(zMeanName)

        return


class drugMiRNA(linkData):
    def __init__(self, args, dataPrefix, embeddedName):
        dataName = 'drugMiRNA'
        super(drugMiRNA, self).__init__(args, dataPrefix, dataName, embeddedName)
        self.XDataLabels, self.XDataDict, self.yLabelsMul = self.getDrugMiRNA()
        return
    def name2ID(self):
        dataPrefix = self.dataPrefix
        dataName = self.dataName

        drugbankName = os.path.join(dataPrefix,dataName+'Data/drugbank_id.csv')
        drugbankName2ID = pd.read_csv(drugbankName)
        drugbankName2ID.columns = ['iname', 'bank_id']
        drugbankName2ID['iname'] = drugbankName2ID['iname'].str.lower()

        cid2DBName = os.path.join(dataPrefix, dataName+'Data/CID2DB.txt')
        cid2DBData = pd.read_csv(cid2DBName, header=0, delimiter='\t')

        drugbankIDInter = drugbankName2ID.loc[drugbankName2ID['bank_id'].isin(cid2DBData['bank_id'])]

        zMeanData = self.zMeanData
        zMeanData['iname'] = zMeanData['iname'].str.lower()

        zMeanInterBank = zMeanData.loc[zMeanData['iname'].isin(drugbankIDInter['iname'])]
        bankInterZMean = drugbankIDInter.loc[drugbankIDInter['iname'].isin(zMeanInterBank['iname'])]

        drugLsName = os.path.join(dataPrefix, dataName+'Data/drug_list.txt')
        drugLs = pd.read_csv(drugLsName, delimiter='\t', header=None)
        drugLs.columns = ['CID', 'SMILES']

        col_name = zMeanInterBank.columns.tolist()
        col_name.insert(1,'bank_id')
        col_name.insert(2,'CID')
        col_name.insert(3,'drugMiRNA_ind')
        zMeanInterBank = zMeanInterBank.reindex(columns=col_name)

        zMeanInterBank = zMeanInterBank.reset_index(drop=True)
        bankInterZMean = bankInterZMean.reset_index(drop=True)

        for ind in zMeanInterBank.index:
            name = zMeanInterBank.loc[ind,'iname']
            bank_id =  bankInterZMean.loc[bankInterZMean['iname']==name,'bank_id'].values[0]
            zMeanInterBank.loc[ind,'bank_id'] =bank_id
            cid = cid2DBData.loc[cid2DBData['bank_id']==bank_id,'CID'].values[0]
            zMeanInterBank.loc[ind, 'CID'] = cid
            zMeanInterBank.loc[ind,'drugMiRNA_ind'] = drugLs[drugLs['CID']==cid].index.values[0]
        return zMeanInterBank
    def getDrugMiRNA(self):
        dataPrefix = self.dataPrefix
        dataName = self.dataName

        # Expression data
        zMeanInterBank = self.name2ID()
        zMeanInterBank.to_csv(os.path.join(self.args.outputFile, 'zMeanInterUsed.csv'))

        # label
        labelName = os.path.join(dataPrefix, dataName+'Data/drug_miRNA.csv')
        labelData = pd.read_csv(labelName, delimiter=',',header=None).values
        labelDataSub = np.array([labelData[int(temp)] for temp in zMeanInterBank['drugMiRNA_ind']])
        np.savetxt(os.path.join(self.args.outputFile, 'drug_miRNA_known.csv'), labelDataSub, fmt='%d')

        print(labelDataSub.shape)
        labelAr = np.array([labelDataSub[i,j] for i in range(labelDataSub.shape[0]) for j in range(labelDataSub.shape[1])])

        drugFeatName = os.path.join(dataPrefix, dataName+'Data/drug_feature_matrix.txt')
        drugFeatFinger = pd.read_csv(drugFeatName, delimiter='\t', header=None).values
        drugFeatFinger = drugFeatFinger[[int(item) for item in zMeanInterBank['drugMiRNA_ind']]]

        miRNAExpressionName = os.path.join(dataPrefix, dataName+'Data/ncrna_expression_full.txt')
        miRNAExpressionData = pd.read_csv(miRNAExpressionName, delimiter='\t', header=None).values

        miRNAGOSimName = os.path.join(dataPrefix, dataName+'Data/ncrna_GOsimilarity_full.txt')
        miRNAGOSimData = pd.read_csv(miRNAGOSimName, delimiter='\t', header=None).values

        # Stack function
        embedStack = lambda x,y: np.hstack((x,y))

        fingerExpX = np.array([embedStack(drugFeatFinger[indI], miRNAExpressionData[indJ]) for indI in range(labelDataSub.shape[0])
                                                                    for indJ in range(labelDataSub.shape[1])])
        fingerGOSimX = np.array([embedStack(drugFeatFinger[indI], miRNAGOSimData[indJ]) for indI in range(labelDataSub.shape[0])
                                                                    for indJ in range(labelDataSub.shape[1])])
        expressExpX = np.array([embedStack(zMeanInterBank.iloc[indI,4:], miRNAExpressionData[indJ]) for indI in range(labelDataSub.shape[0])
                                                                    for indJ in range(labelDataSub.shape[1])])
        expressGOSimX = np.array([embedStack(zMeanInterBank.iloc[indI,4:], miRNAGOSimData[indJ]) for indI in range(labelDataSub.shape[0])
                                                                    for indJ in range(labelDataSub.shape[1])])

        XDataLabels = ['FingerGOSim', 'FingerExp', 'ExpressionGOSim', 'ExpressionExp']
        XDataDict = {'FingerExp': fingerExpX, 'FingerGOSim': fingerGOSimX,
                     'ExpressionExp': expressExpX, 'ExpressionGOSim': expressGOSimX}

        yLabelsMul = labelAr.reshape(-1,1)

        return XDataLabels, XDataDict, yLabelsMul



def getData(args, dataPrefix, dataName, embeddedName):
    outputFile = args.outputFile = 'output/'+dataName+'Data/'
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    
    if dataName=='drugMiRNA':
        dataObject = drugMiRNA(args, dataPrefix, embeddedName)
    else:
        sys.exit('Data name ('+dataName+') is wrong!')

    return dataObject



def getArgs():
    parser = argparse.ArgumentParser(description='Get link data')
    parser.add_argument('--expName', default='processLinkData')
    
    parser.add_argument('--dataPrefix', default='../../dataset/')
    parser.add_argument('--dataName', default='drugDis', choices=['drugMiRNA'])
    parser.add_argument('--embeddedName', default='EmbeddedZLast_mean.csv', type=str, choices=['EmbeddedZLast_mean.csv'])
    
    parser.add_argument('--dimPara', default=[128, 32], type=int, nargs='+')
    
    args = parser.parse_args()
    return args



def linkMain():
    args = getArgs()
    dataPrefix = args.dataPrefix
    dataName = args.dataName
    embeddedName = args.embeddedName

    dataObj = getData(args, dataPrefix, dataName, embeddedName)
    return dataObj



if __name__ == '__main__':
    dataObj = linkMain()



