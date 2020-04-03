
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.signal

version = '0'
Obj = 'cyl35_red_plan'

class transition_experience():
    path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/'
    dest_path = '/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/data/dataset_processed/' 

    def __init__(self, Load=True, discrete = True, postfix='', Object = Obj, with_fingers = False):

        if discrete:
            self.mode = 'd' # Discrete actions
        else:
            self.mode = 'c' # Continuous actions

        self.Object = Object
        self.with_fingers = with_fingers
        
        self.postfix = postfix
        if postfix == 'bu':
            self.file_name = self.path + 'internal/' + 'raw_' + self.Object + '_' + self.mode + '_v' + version + self.postfix + '.obj'
        else:
            self.file_name = self.path + 'raw_' + self.Object + '_' + self.mode + '_v' + version + self.postfix + '.obj'

        if Load:
            self.load()
        else:
            self.clear()
        
    def add(self, time, state, action, next_state, done):
        self.memory += [(time, state, action, next_state, done)]
        
    def clear(self):
        self.memory = []

    def load(self):
        if os.path.isfile(self.file_name):
            print('Loading data from ' + self.file_name)
            with open(self.file_name, 'rb') as filehandler:
                self.memory = pickle.load(filehandler)
            print('Loaded transition data of size %d.'%self.getSize())
        else:
            self.clear()

    def getComponents(self):

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])

        return states, actions, next_states

    def save(self):
        print('Saving data...')
        file_pi = open(self.file_name, 'wb')
        pickle.dump(self.memory, file_pi)
        print('Saved transition data of size %d.'%self.getSize())
        file_pi.close()

    def getSize(self):
        return len(self.memory)

    def plot_data(self):

        # del self.memory[86681]
        # file_pi = open(self.file_name, 'wb')
        # pickle.dump(self.memory, file_pi)

        T = np.array([item[0] for item in self.memory])
        states = np.array([item[1] for item in self.memory])

        states[:,:2] *= 1000.
        states[:,3:-2] *= 1000.
        done = np.array([item[4] for item in self.memory])
        failed_states = states[done]

        
        

        plt.figure(1)
        ax1 = plt.subplot(221)
        # ax1.plot(states[:,0],states[:,1],'-k')
        ax1.plot(states[:,3],states[:,4],'.b')
        ax1.plot(states[:,5],states[:,6],'.b')
        ax1.plot(states[:,7],states[:,8],'.b')
        ax1.plot(states[:,9],states[:,10],'.b')
        ax1.plot(states[:,0],states[:,1],'.y')
        ax1.plot(failed_states[:,0],failed_states[:,1],'.r')
        ax1.set(title='Object position')
        ax1.axis('equal')
        plt.ylim((-50, 140))
        plt.xlim((-60, 130))
        
        ax2 = plt.subplot(222)
        ax2.plot(states[:,-2],states[:,-1],'.k')
        ax2.plot(failed_states[:,-2],failed_states[:,-1],'.r')
        ax2.set(title='Actuator loads')
        ax2.axis('equal')
        plt.xlim((-10, 300))
        plt.ylim((-300, 10))

        ax3 = plt.subplot(223)
        ax3.plot(np.rad2deg(states[:,2]),'-k')
        ax3.set(title='Object angle')
        plt.ylim((-180., 180.))

        plt.show()

    def save_to_file(self):

        filen = self.path + 'transition_data_' + self.mode + '.db'

        n = self.getSize()

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        inx = np.where(done)

        M = np.concatenate((states, actions, next_states), axis=1)
        M = np.delete(M, inx, 0)

        np.savetxt(filen, M, delimiter=' ')

    def transform_angles(self, angles):
        if self.Object == 'sqr30' or self.Object == 'rec60':
             return angles % (np.pi/2.)
        if self.Object == 'poly6':
             return angles % (np.pi/3.)
        if self.Object == 'poly10':
             return angles % (np.pi/5.)
        if self.Object == 'elp40' or self.Object == 'rec60' or self.Object == 'rec10' or self.Object == 'cre55':
             return angles % (np.pi)
        if self.Object == 'str40':
             return angles % (2*np.pi/6.) #!!!!!!!!!!!
        if self.Object == 'tri50':
             return angles % (2*np.pi/3.)
        return angles

    def validate_drops(self, states, done):
        for i in range(states.shape[0]-1):
            if np.linalg.norm(states[i,:2]-states[i+1,:2]) > 8. and not done[i]:
                done[i] = True
        
        return done
    
    def process_transition_data(self, stepSize = 1, plot = False):

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

        def new_clean(D, done):
            
            episodes = []

            ks = 0
            kf = 1
            while kf < D.shape[0]:
                if kf >= D.shape[0]:
                    break 

                # Idenfify end of episode
                while kf < D.shape[0]-1 and not done[kf]:
                    kf += 1

                if kf - ks < 15:
                    D, done = Del(D, done, range(ks, kf+1))
                    kf = ks + 1
                    continue

                # Avoid the peaks at grasp
                # ic = 0
                # for i in range(ks+20, ks+1, -1):
                #     ic += 1
                #     if np.any(np.abs(D[i, 2:4]-D[i-1,2:4]) > 40.):
                #         D, done = Del(D, done, range(ks, ks+21-ic))
                #         kf -= 21-ic
                #         break

                while np.linalg.norm(D[kf,:2]-D[kf-1,:2]) > 1.2 or np.linalg.norm(D[kf,:2]-D[kf,self.state_action_dim:self.state_action_dim+2]) > 1.2:
                    D, done = Del(D, done, kf)
                    kf -= 1

                # Apply filter to episode
                h = [40, 40, 100, 100]
                for i in range(self.state_dim):
                    if i == 4:
                        continue
                    try:
                        D[ks:kf,i] = medfilter(D[ks:kf,i], h[i])
                    except:
                        D[ks:kf,i] = medfilter(D[ks:kf,i], 40)

                # Update next state columns
                D[ks:kf, self.state_action_dim:] = D[ks+1:kf+1, :self.state_dim]
                episodes.append(D[ks:kf,:])
                D, done = Del(D, done, kf)

                ks = kf
                kf += 1

            i = 0
            while i < D.shape[0]:
                if np.linalg.norm(D[i,:2]-D[i,self.state_action_dim:self.state_action_dim+2]) > 1.2 or D[i,0] < -70. or D[i,0] > 120 or D[i,self.state_action_dim] < -70. or D[i,self.state_action_dim] > 120:
                    D, done = Del(D, done, i)
                else:
                    i += 1

            return D, done, episodes

        def multiStep(D, done, stepSize): 
            Dnew = []
            ia = range(self.state_dim,self.state_dim+self.action_dim)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize, ia]) or np.any(done[i:i+stepSize]):
                    continue

                Dnew.append( np.concatenate((D[i,:ia[0]], a, D[i+stepSize-1, ia[-1]+1:]), axis=0) )

            return np.array(Dnew)

        print('[transition_experience] Saving transition data...')
        is_start = 1
        is_end = 277

        states = np.array([item[1] for item in self.memory])
        states[:,:2] *= 1000.
        actions = np.array([item[2] for item in self.memory])
        next_states = np.array([item[3] for item in self.memory])
        next_states[:,:2] *= 1000.
        
        done = np.array([item[4] for item in self.memory]) 

        # Explot symmetry of object profile
        states[:,2] = self.transform_angles(states[:,2])
        next_states[:,2] = self.transform_angles(next_states[:,2])

        if np.any(self.Object == np.array(['sqr30','poly10','poly6','elp40','str40','tri50','rec60','rec10','cre55','sem60'])): # Include orientation angle
            if self.with_fingers:
                states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
                next_states = next_states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
                states[:,5:] *= 1000.
                next_states[:,5:] *= 1000.
            else:
                states = states[:,[0,1,11,12,2]]
                next_states = next_states[:,[0,1,11,12,2]]
        else:
            if self.with_fingers:
                states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
                next_states = next_states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
                states[:,4:] *= 1000.
                next_states[:,4:] *= 1000.
            else:
                states = states[:,[0, 1, 11, 12]]
                next_states = next_states[:,[0, 1, 11, 12]]

        # For data from recorder
        if self.postfix != 'bu':
            next_states = np.roll(states, -1, axis=0) 

        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1] + self.state_dim
        self.state_action_dim = self.state_dim + self.action_dim 

        done = self.validate_drops(states, done)
        if np.any(self.Object == np.array(['poly6', 'cyl35_red_plan'])): # When the actions length is not at the same size as the states
            done[-1] = True

        # Add grasp state to action
        inx = np.where(done)[0]
        inx = np.insert(inx, 0, -1)
        AF = np.zeros((1,self.action_dim))
        j = 0
        for i in range(1,len(inx)):
            grasp_state = states[inx[i-1]+1,:]
            As = np.tile(grasp_state, (inx[i]-inx[i-1], 1))
            af = np.concatenate((actions[inx[i-1]+1:inx[i]+1], As), axis=1)
            j += af.shape[0]
            AF = np.append(AF, af, axis=0)
        actions = AF[1:,:]

        D = np.concatenate((states, actions, next_states), axis = 1)

        # Remove false drops when motion is continuous
        for i in range(len(done)-1):
            if done[i]:
                if np.linalg.norm(states[i,:2]-states[i+1,:2]) < 3.:
                    done[i] = False

        # Start dist.
        St = [D[0,:self.state_dim]]
        for i in range(1, D.shape[0]-1):
            if done[i-1] and not done[i]:
                St.append(D[i,:self.state_dim])
        St = np.array(St)
        s_start = np.mean(St, 0)
        s_std = np.std(St, 0)
        print "start mean: ", s_start
        print "start std.: ", s_std

        D, done, episodes = new_clean(D, done)

        # path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'
        # ix = [ 6, 40, 83, 89, 94, 95, 96, 101, 107, 109]
        # version = 1
        # test_paths = []
        # action_seq = []
        # Suc = [ ]
        # for i in ix:
        #     e = episodes[i]
        #     plt.plot(e[:,0],e[:,1],'-')
        #     test_paths.append(e[:,:13])
        #     action_seq.append(e[:,13:15])
        #     Suc.append(False)
        # with open(path + 'testpaths_' + self.Object + '_c_v' + str(version) + '.pkl', 'w') as f: 
        #     pickle.dump([action_seq, test_paths, 'cyl35', Suc], f)
        # plt.show()

        # i = 0
        # j = 0
        # for e in episodes:
        #     if e.shape[0] < 700:
        #         i += 1
        #         continue
        #     plt.plot(D[:,0], D[:,1], '.y')
        #     plt.plot(e[:,0], e[:,1],'-k')
        #     print i, j
        #     i += 1
        #     j += 1
        #     plt.show()

        # exit(1)       

        with open(self.dest_path + 't42_' + self.Object + '_data_discrete_v' + version + '.1_d' + str(states.shape[1]) + '_m' + str(stepSize) + '_episodes.obj', 'wb') as f: 
            pickle.dump(episodes, f)
        print len(episodes)
        exit(1)

        if stepSize > 1:
            D = multiStep(D, done, stepSize)

        # for i in range(done.shape[0]):
        #     if done[i]:
        #         done[i-2:i] = True

        inx = np.where(done)[0]
        D = np.delete(D, inx, 0) # Remove drop transitions
        done = np.delete(done, inx, 0)

        self.D = D

        # Bounds
        print "Max: ", np.max(D, 0)[:self.state_dim]
        print "Min: ", np.min(D, 0)[:self.state_dim]

        is_start = 60000
        is_end = is_start+100

        with open(self.dest_path + 't42_' + self.Object + '_data_discrete_v' + version + '_d' + str(states.shape[1]) + '_m' + str(stepSize) + '.obj', 'wb') as f: 
            pickle.dump([D, self.state_dim, self.action_dim, is_start, is_end], f)
        print "Saved processed data file with " + str(D.shape[0]) + " transition points."

        if plot:
            plt.figure(0)
            plt.plot(D[:,0], D[:,1],'.-k')
            # plt.plot(D[:,4],'.-k')
            # for _ in range(1000):
            #     j = np.random.randint(D.shape[0])
            #     plt.plot([D[j,0], D[j,6]], [D[j,1], D[j,7]],'o-r')
            plt.title(str(stepSize))
            plt.show()
    
    def process_svm(self, stepSize = 1):

        def clean_done(states, done):
            # Cancel drop mark if episode is very short
            i = 0
            while i < states.shape[0]-1:
                j = i + 1
                while j < states.shape[0] and not done[j]:
                    j +=1
                j = min(j, states.shape[0]-1)
                if done[j] and j-i < 10:
                    done[j] = False 
                i = j

            # Cancel drop if load is not critical
            # for i in range(states.shape[0]):
            #     if done[i] and np.all(np.abs(states[i, 2:4]) < 260) and np.all(np.abs(states[i, 2:4]) > 40):
            #         # if np.random.uniform() > 0.5:
            #         done[i] = False

            return done

        def multiStep(D, done, stepSize): 
            Dnew = []
            done_new = []
            ia = range(self.state_dim,self.state_dim+self.action_dim)
            for i in range(D.shape[0]-stepSize):
                a = D[i, ia] 
                if not np.all(a == D[i:i+stepSize, ia]):
                    continue
                
                if np.any(done[i:i+stepSize]):
                    done_new.append(True)
                else:
                    done_new.append(False)

                Dnew.append( np.concatenate((D[i,:ia[0]], a), axis=0) )

            return np.array(Dnew), np.array(done_new)

        from sklearn import svm
        from sklearn.preprocessing import StandardScaler
        
        states = np.array([item[1] for item in self.memory])
        states[:,:2] *= 1000.
        actions = np.array([item[2] for item in self.memory])
        done = np.array([item[4] for item in self.memory])

        # Explot symmetry of object profile
        states[:,2] = self.transform_angles(states[:,2])

        if np.any(self.Object == np.array(['sqr30','poly10','poly6','elp40','str40','tri50','rec60','rec10','cre55','sem60'])): # Include orientation angle
            if self.with_fingers:
                states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
                states[:,5:] *= 1000.
            else:
                states = states[:,[0, 1, 11, 12, 2]]
            # states[:,4] = np.sin(states[:,4])/np.cos(states[:,4])
        else:
            if self.with_fingers:
                states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
                states[:,4:] *= 1000.
            else:
                states = states[:,[0, 1, 11, 12]]

        done = self.validate_drops(states, done)
        if self.Object == 'poly6':
            done[-1] = True

        # Add grasp state to action
        inx = np.where(done)[0]
        inx = np.insert(inx, 0, -1)
        AF = np.zeros((1,states.shape[1]+2))
        for i in range(1,len(inx)):
            grasp_state = states[inx[i-1]+1,:]
            As = np.tile(grasp_state, (inx[i]-inx[i-1], 1))
            af = np.concatenate((actions[inx[i-1]+1:inx[i]+1], As), axis=1)
            AF = np.append(AF, af, axis=0)
        actions = AF[1:,:]
            
        # Remove false drops when motion is continuous
        for i in range(len(done)-1):
            if done[i]:
                if np.linalg.norm(states[i,:2]-states[i+1,:2]) < 3.:
                    done[i] = False

        done = clean_done(states, done)

        for i in range(done.shape[0]):
            if done[i]:
                done[i-3:i] = True

        SA = np.concatenate((states, actions), axis=1)
        if stepSize > 1:
            SA, done = multiStep(SA, done, stepSize)
        print('Transition data with steps size %d has now %d points'%(stepSize, SA.shape[0]))

        inx_fail = np.where(done)[0]
        # inx_fail = inx_fail[np.random.choice(inx_fail.shape[0], 1000, replace=False)]
        print "Number of failed states " + str(inx_fail.shape[0])
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], inx_fail.shape[0], replace=False)]
        SA = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)
        
        # with open(self.dest_path + 't42_' + self.Object + '_svm_data_' + self.mode + '_v' + version + '_d' + str(states.shape[1]) + '_m' + str(stepSize) + '.obj', 'wb') as f: 
        #     pickle.dump([SA, done], f)
        # print('Saved svm data.')

        ##  Test data
        # Normalize
        scaler = StandardScaler()
        SA = scaler.fit_transform(SA)
        x_mean = scaler.mean_
        x_std = scaler.scale_

        # Test data
        ni = int(np.floor(0.1*inx_fail.shape[0]))
        T = np.where(done)[0]
        inx_fail = T[np.random.choice(T.shape[0], ni, replace=False)]
        T = np.where(np.logical_not(done))[0]
        inx_suc = T[np.random.choice(T.shape[0], ni, replace=False)]
        SA_test = np.concatenate((SA[inx_fail], SA[inx_suc]), axis=0)
        done_test = np.concatenate((done[inx_fail], done[inx_suc]), axis=0)

        SA = np.delete(SA, inx_fail, axis=0)
        SA = np.delete(SA, inx_suc, axis=0)
        done = np.delete(done, inx_fail, axis=0)
        done = np.delete(done, inx_suc, axis=0)

        print 'Fitting SVM...'
        clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
        clf.fit( list(SA), 1*done )
        print 'SVM fit with %d classes: '%len(clf.classes_) + str(clf.classes_)

        s = 0
        s_suc = 0; c_suc = 0
        s_fail = 0; c_fail = 0
        for i in range(SA_test.shape[0]):
            p = clf.predict_proba(SA_test[i].reshape(1,-1))[0]
            fail = p[1]>0.5
            # print p, done_test[i], fail
            s += 1 if fail == done_test[i] else 0
            if done_test[i]:
                c_fail += 1
                s_fail += 1 if fail else 0
            else:
                c_suc += 1
                s_suc += 1 if not fail else 0
        print 'Success rate: ' + str(float(s)/SA_test.shape[0]*100)
        print 'Drop prediction accuracy: ' + str(float(s_fail)/c_fail*100)
        print 'Success prediction accuracy: ' + str(float(s_suc)/c_suc*100)
        # exit(1)

        # ######################################################
        # More classifiers
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.gaussian_process.kernels import RBF
        

        names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Gaussian Process',
         'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost']

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
            clf.fit(list(SA), 1*done)
            score = clf.score(SA_test, 1*done_test)
            scores.append(score)
            print name, score
            print SA_test[0].shape, SA_test[0]
            print clf.predict(SA_test[0].reshape(1,-1)), done_test[0]

        scores = dict( zip( names, scores))
            
        # ######################################################

        return np.array([float(s)/SA_test.shape[0]*100, float(s_fail)/c_fail*100, float(s_suc)/c_suc*100]), scores



