#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os.path

class svm_failure():

    r = 0.1
    path = '/home/pracsys/catkin_ws/src/t42_control/nn_node/models/'
    dim_ = 4
    data_version_ = 0
    stepSize_ = 1

    def __init__(self, simORreal = 't42_cyl35', discrete = True):

        self.mode = 'd' if 'discrete' else 'c'
        self.simORreal = simORreal

        if np.any(simORreal == np.array(['t42_sqr30','t42_poly10','t42_poly6','t42_elp40','t42_tri50','t42_egg50','t42_str40','t42_rec60','t42_rec10','t42_cre55','t42_sem60'])): # Include orientation angle
            self.dim_ += 1

        self.load_data()

        print 'All set!'

    def load_data(self):

        self.postfix = '_v' + str(self.data_version_) + '_d' + str(self.dim_) + '_m' + str(self.stepSize_)
        if os.path.exists(self.path + self.simORreal + '_svm_fit_discrete' + self.mode + self.postfix + '.obj'):
            with open(self.path + self.simORreal + '_svm_fit_discrete' + self.mode + self.postfix + '.obj', 'rb') as f: 
                self.clf, self.x_mean, self.x_std = pickle.load(f)
            print('[SVM] Loaded svm fit.')
        else:
            File = self.simORreal + '_svm_data_' + self.mode +  self.postfix + '.obj' # <= File name here!!!!!

            print('[SVM] Loading data from ' + File)
            with open(self.path + File, 'rb') as f: 
                self.SA, self.done = pickle.load(f)
            print('[SVM] Loaded svm data.')            

            # Normalize
            scaler = StandardScaler()
            self.SA = scaler.fit_transform(self.SA)
            self.x_mean = scaler.mean_
            self.x_std = scaler.scale_
        
            print 'Fitting SVM...'
            self.clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
            self.clf.fit( list(self.SA), 1*self.done )

            with open(self.path + self.simORreal + '_svm_fit_discrete' + self.mode +  self.postfix + '.obj', 'wb') as f: 
                pickle.dump([self.clf, self.x_mean, self.x_std], f)

        print 'SVM ready with %d classes: '%len(self.clf.classes_) + str(self.clf.classes_)

    def probability(self, s, a):

        sa = np.concatenate((s,a), axis=0).reshape(1,-1)

        # Normalize
        sa = (sa - self.x_mean) / self.x_std

        p = self.clf.predict_proba(sa)[0][1]

        return p#, self.clf.predict(sa)

