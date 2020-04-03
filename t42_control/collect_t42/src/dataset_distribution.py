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
sys.path.insert(0, '/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/src/')
from data_load import data_load
from svm_class import svm_failure

np.random.seed(415)

path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'
transition_path = '/home/pracsys/Dropbox/transfer/RUM/Model_T42/transition_data/'
failure_path = '/home/pracsys/Dropbox/transfer/RUM/Model_T42/failure_data/'


def get_objs(discrete=True):
    files = glob.glob(path + "*.obj")

    O = []
    F = []
    for file in files:
        if file.find('_c_' if discrete else '_d_') > 0:
            continue
        obj = file[file.find('raw_')+4:file.find('_d_' if discrete else '_c_')]
        O.append(obj)
        F.append(file)

    return O, F

def check_exist(target_file, cd_str, path):
    files = glob.glob(path + cd_str + "*.obj")

    for file in files:
        if file.find(target_file) > 0:
            return True

    return False

def process_transition_data(Object, memory, red=False):

    def medfilter(x, W):
        w = int(W/2)
        x_new = np.copy(x)
        for i in range(1, x.shape[0]-1):
            if i < w:
                x_new[i] = np.mean(x[:i+w])
            elif i > x.shape[0]-w:
                x_new[i] = np.mean(x[i-w:])
            else:
                x_new[i] = np.mean(x[i-w:i+w])
        return x_new

    def Del(D, done, inx):
        D = np.delete(D, inx, 0)
        done = np.delete(done, inx, 0)

        return D, done

    def validate_drops(states, done):
        for i in range(states.shape[0]-1):
            if np.linalg.norm(states[i,:2]-states[i+1,:2]) > 8. and not done[i]:
                done[i] = True

        return done

    def new_clean(D, done, state_action_dim, state_dim):
        
        episodes = []

        ks = 0
        kf = 1
        while kf < D.shape[0]:
            if kf >= D.shape[0]:
                break 

            # Identify end of episode
            while kf < D.shape[0]-1 and not done[kf]:
                kf += 1

            if kf - ks < 50:
                D, done = Del(D, done, range(ks, kf+1))
                kf = ks + 1
                continue

            while np.linalg.norm(D[kf,1:3]-D[kf-1,1:3]) > 1.2 or np.linalg.norm(D[kf,1:3]-D[kf,state_action_dim+1:state_action_dim+2+1]) > 1.2:
                D, done = Del(D, done, kf)
                kf -= 1

            # Update next state columns
            D[ks:kf, state_action_dim+1:] = D[ks+1:kf+1, 1:state_dim+1]
            episodes.append(D[ks:kf,:])
            D, done = Del(D, done, kf)

            ks = kf
            kf += 1

        i = 0
        while i < D.shape[0]:
            if np.linalg.norm(D[i,1:3]-D[i,state_action_dim+1:state_action_dim+2+1]) > 1.2 or D[i,1] < -70. or D[i,1] > 120 or D[i,state_action_dim+1] < -70. or D[i,state_action_dim+1] > 120:
                D, done = Del(D, done, i)
            else:
                i += 1

        return D, done, episodes

    print('Saving transition data...')

    T = np.array([item[0] for item in memory])
    states = np.array([item[1] for item in memory])
    actions = np.array([item[2] for item in memory])
    next_states = np.array([item[3] for item in memory])
    done = np.array([item[4] for item in memory]) 

    if red:
        actions = np.fliplr(actions)
        print('Flipped.')

    states[:,:2] *= 1000.
    states[:,3:11] *= 1000.
    next_states[:,:2] *= 1000.
    next_states[:,3:11] *= 1000.

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    state_action_dim = state_dim + action_dim 

    done = validate_drops(states, done)
    # if np.any(Object == np.array(['poly6', 'cyl35_red'])): # When the actions length is not at the same size as the states
    #     done[-1] = True

    D = np.concatenate((T.reshape(-1,1), states, actions, next_states), axis = 1)

    # Remove false drops when motion is continuous
    for i in range(len(done)-1):
        if done[i]:
            if np.linalg.norm(states[i,:2]-states[i+1,:2]) < 3.:
                done[i] = False

    D, done, episodes = new_clean(D, done, state_action_dim, state_dim)

    E = []
    i = 0
    s = 0
    for e in episodes:
        et = {'episode_number': i, 'time_stamps': e[:,0], 'states': e[:,1:14], 'actions': e[:,14:16], 'next_states': e[:,16:]}
        i += 1
        E.append(et)
        s += e.shape[0]
    print 'For object ' + Object + ', saved ' + str(s) + ' points in ' + str(i) + ' episodes.'

    return E, s       

def main():
    discrete = True # discrete or continuous
    objs, files = get_objs(discrete)

    cd_mode = '_d_' if discrete else '_c_'

    # Process raw data
    if 1:
        for obj, file in zip(objs, files):
            
            target_file = 'raw_t42_' + obj + cd_mode[:-1] + '.obj'
            if not check_exist(target_file, ('discrete/' if discrete else 'continuous/'), transition_path):
                print('Loading data for ' + obj + '...')
                try:
                    with open(file, 'rb') as filehandler:
                        memory = pickle.load(filehandler)
                except:
                    print('Failed to load ' + file)
                    continue
                print('Loaded transition data of size %d.'%len(memory))

                # if len(memory) < 100000:
                #     continue

                #### Transition data ####
                episodes, data_size = process_transition_data(obj, memory, True if target_file.find('_red') > 0 else False)

                # Each episode is a dictionary with keys {'episode_number', 'time_stamps', 'states', 'actions', 'next_states'}
                with open(transition_path + ('discrete/' if discrete else 'continuous/') + target_file, 'wb') as f: 
                    pickle.dump(episodes, f)
                print 'Saved transition data for object ' + obj + '.'
            else:
                with open(transition_path + ('discrete/' if discrete else 'continuous/') + target_file, 'rb') as f: 
                    episodes = pickle.load(f)

            if obj.find('_red') > 0: # No need for failure data for the red hand
                continue

            #### Failure ####
            target_file = 'classifier_data_t42_' + obj + cd_mode[:-1] + '.obj'
            if not check_exist(target_file, ('discrete/' if discrete else 'continuous/'), failure_path):
                test_inx = np.random.choice(len(episodes), size=50, replace = False)
                SA_train = []
                SA_test = []
                label_train = []
                label_test = []
                A_seq = []

                for i in range(len(episodes)):
                    e = episodes[i]

                    if np.any(i == test_inx):
                        sa_bad = np.concatenate((e['states'][-1,:], e['actions'][-1,:]), axis=0)
                        SA_test.append(sa_bad)
                        label_test.append(True)
                        
                        j_good = np.random.randint(e['states'].shape[0]-10)
                        sa_good = np.concatenate((e['states'][j_good,:], e['actions'][j_good,:]), axis=0)
                        SA_test.append(sa_good)
                        label_test.append(False)

                        A_seq.append(e['actions'])
                    else:
                        sa_bad = np.concatenate((e['states'][-3:,:], e['actions'][-3:,:]), axis=1)
                        for sa in sa_bad:
                            SA_train.append(sa)
                            label_train.append(True)
                        
                        for _ in range(3):
                            j_good = np.random.randint(e['states'].shape[0]-10)
                            sa_good = np.concatenate((e['states'][j_good,:], e['actions'][j_good,:]), axis=0)
                            SA_train.append(sa_good)
                            label_train.append(False)

                et = {'train_data': SA_train, 'train_labels': label_train, 'test_data': SA_test, 'test_labels': label_test, 'test_action_seq': A_seq}
                with open(failure_path + ('discrete/' if discrete else 'continuous/') + target_file, 'wb') as f: 
                    pickle.dump(et, f)
                print "Saved %d training points and %d test points for object %s."%(len(SA_train), len(SA_test), obj)

                if 0: # Test data
                    # Evaluate classifiers
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
                    from sklearn.svm import SVC
                    from sklearn.neural_network import MLPClassifier
                    from sklearn.gaussian_process.kernels import RBF

                    names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Gaussian Process', 'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost']
                    classifiers = [
                        KNeighborsClassifier(3),
                        SVC(kernel="linear", C=0.025),
                        SVC(gamma=2, C=1),
                        GaussianProcessClassifier(1.0 * RBF(1.0)),
                        DecisionTreeClassifier(max_depth=5),
                        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        MLPClassifier(alpha=1, max_iter=1000),
                        AdaBoostClassifier()]

                    # iterate over classifiers
                    scores = []
                    for name, clf in zip(names, classifiers):
                        clf.fit(SA_train, 1*label_train)
                        score = clf.score(SA_test, 1*label_test)
                        scores.append(score)
                        print name, score

                    scores = dict( zip( names, scores))
                
    return 1


if __name__=='__main__':
    main()