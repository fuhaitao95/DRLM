#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:12:05 2022

@author: fht
"""

# python experiments.py --expName preprocess


import argparse
import sys


def mainExp():
    # args = getArgs()
    # choices = ['representationLearning', 'processLinkData', 'expCV']
    expName = sys.argv[2]
    print(expName)

    result = dict()
    if (expName == 'representationLearning'):
        from representationLearning import representOperator
        result['z_mean'], result['label_contain'] = representOperator()

    elif (expName == 'expCV'):
        from expCV import CVOperator
        result['resultCV'] = CVOperator()

    else:
        sys.exit('Experiment name is wrong!')
    return result


if __name__ == '__main__':
    result = mainExp()
