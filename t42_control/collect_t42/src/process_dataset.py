#!/usr/bin/env python

import numpy as np
import time
import random
import pickle
import os.path
import matplotlib.pyplot as plt
from transition_experience import *
import glob

import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/t42_control/gpup_gp_node/src/')
from data_load import data_load
from svm_class import svm_failure

path = '/home/juntao/catkin_ws/src/t42_control/hand_control/data/dataset/'
dest_path = '/home/juntao/catkin_ws/src/t42_control/gpup_gp_node/data/dataset_processed/'

def get_objs(discrete=True):
    files = glob.glob(path + "*.obj")

    F = []
    for file in files:
        if file.find('_c_' if discrete else '_d_') > 0:
            continue
        obj = file[file.find('raw_')+4:file.find('_d_' if discrete else '_c_')]
        F.append(obj)

    return F

def check_opt(self, obj):
    files = glob.glob(dest_path + "*.obj")

    for file in files:
        if file.find('t42_' + obj + '_opt') > 0:
            return True

    return False


def main():
    discrete = True
    objs = get_objs(discrete)
    # objs = ['cyl35','cyl45','cyl30','str40']
    # objs = ['poly6','poly10','elp40','sqr30']

    # Process raw data
    if 0:
        download_dir = dest_path + '/summary.csv' 
        csv = open(download_dir, "w") 
        # csv.write('name, Success rate, fail accuracy, success accuracy \n')
            
        i = 0
        for obj in objs:
            if np.any(obj == np.array(['cyl45_right','cyl30','cyl35','sqr30','poly6'])):
                continue
            print("\n\n[process_dataset] Processing object '%s'...\n\n"%obj)
            texp = transition_experience(Load = True, discrete=discrete, postfix='', Object = obj, with_fingers = False)

            # texp.process_transition_data(stepSize = 1, plot = False)
            O, scores = texp.process_svm(stepSize = 1)

            if i == 0:
                csv.write('Object, ')
                for key in scores.keys():
                    csv.write(key + ', ')
                csv.write('\n')
                i += 1

            csv.write(obj + ', ')
            for v in scores.values():
                csv.write(str(v) + ', ')
            csv.write('\n')

            # csv.write(obj + ',' + str(O[0]) + ',' + str(O[1]) + ',' + str(O[2]) + '\n')

    # Generate kd-trees and pre-compute hyper-parameters
    if 1:
        for i in range(1):
            # with_fingers = False if i == 0 else True
            for obj in objs:
                if np.any(obj == np.array(['cyl45_right','cyl30','cyl35','sqr30','poly6'])):
                    continue
                print("\n\n[process_dataset] Computing hyper-parameters for object '%s'...\n\n"%obj)
                DL = data_load(simORreal = 't42_' + obj, discreteORcont = ('discrete' if discrete else 'cont'), K = 1000, K_manifold = 100, sigma=2.0, dim = 3, dr = 'diff', with_fingers = False)    

                # SVM = svm_failure(simORreal = 't42_' + obj, discrete = discrete)   
    return 1


if __name__=='__main__':
    main()