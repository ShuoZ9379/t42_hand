#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
import time
import glob,os,sys
#rosrun path/to/corl_rollout_real_red.py

def check_outlier(pos,checks):
    if checks[0,:].any():
        if checks.shape[0]==2:
            checks[1,0]=1.
        else:
            if checks[2,:].any():
                checks[1,0]=1.
            elif np.linalg.norm(pos[1,:]-pos[2,:])*1000>1.2:
                checks[1,0]=1.
    else:
        if np.linalg.norm(pos[1,:]-pos[0,:])*1000>1.2:
            checks[1,0]=1.
    return checks

def remove_outlier(states,checks):
    pos=states[:,:2]
    for i in range(1,pos.shape[0]):
        if not checks[i,:].any():
            checks[i-1:i+2,:]=check_outlier(pos[i-1:i+2,:],checks[i-1:i+2,:])
    return checks

def check_bad_end(indices_arr,pos,checks,interval):
    to_check=pos[indices_arr[-interval-1:],:]
    diff=np.roll(to_check,-1,axis=0)-to_check
    diff=diff[:interval,:]
    valid=np.linalg.norm(diff,axis=1)*1000<=1.2
    for i in range(interval):
        if not valid[i]:
            checks[indices_arr[-interval+i:],0]=np.ones(interval-i)
            return checks
    return checks

def remove_end(states,checks,interval):
    pos=states[:,:2]
    valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int))==3
    indices_arr=np.where(valid_idx_ls==True)[0]
    if indices_arr.shape[0]!=0:
        checks=check_bad_end(indices_arr,pos,checks,interval)
    return checks


def medfilter(states,checks,filter_size):
    valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int))==3
    for j in range(4) :
        x=states[:,j]
        x_new=np.copy(x)
        w = int(filter_size/2)
        for i in range(0, x.shape[0]):
            if i < w:
                if valid_idx_ls[:i+w].any():
                    x_new[i] = np.mean(x[:i+w][valid_idx_ls[:i+w]])
            elif i > x.shape[0]-w:
                if valid_idx_ls[i-w:].any():
                    x_new[i] = np.mean(x[i-w:][valid_idx_ls[i-w:]])
            else:
                if valid_idx_ls[i-w:i+w].any():
                    x_new[i] = np.mean(x[i-w:i+w][valid_idx_ls[i-w:i+w]])
        states[:,j]=x_new
    return states

def get_transformed_states(states,rmat,tvec,with_med_filter=True,with_remove_d=True,remove_end_interval=10,filter_size=20):
    transformed_states=np.zeros((states.shape[0],4))
    transformed_states[:,1]=-rmat.T.dot((states[:,12:15]-tvec).T).T[:,0]
    transformed_states[:,0]=rmat.T.dot((states[:,12:15]-tvec).T).T[:,1]
    for i in range(2,4):
        transformed_states[:,i]=states[:,i+28]
    checks=states[:,-3:]

    checks=remove_outlier(transformed_states,checks)

    if with_remove_d:
        checks=remove_end(transformed_states,checks,remove_end_interval)

    transformed_states[:,:2]=transformed_states[:,:2]*1000

    if with_med_filter:
        transformed_states=medfilter(transformed_states,checks,filter_size)

    return np.concatenate((transformed_states,checks),axis=1)

def get_real_states(SS):
    states,checks=np.split(SS,[-3],axis=1) 
    valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int))==3
    return states[valid_idx_ls],valid_idx_ls


rollout = 1
comp = 'szhang'
with_med_filter=True
with_remove_d=True
remove_end_interval=10
filter_size=20

mode_ls=['astar','policy']
ah_with_goal_loc=1
for mode in mode_ls:
    if mode == 'astar':
        Sets = ['_set']
    elif mode == 'policy':
        if ah_with_goal_loc:
            Sets = ['_withgoalloc_set']
        else:
            Sets = ['_nogoalloc_set']
    else:
        continue

    ############################# Rollout ################################
    if rollout:
        import rospy
        from std_srvs.srv import Empty, EmptyResponse
        from rollout_t42.srv import rolloutReq

        rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
        rollout_v2_srv = rospy.ServiceProxy('/rollout/rollout_v2', rolloutReq)
        rospy.init_node('run_rollout_set', anonymous=True)
        state_dim = 35


        for Set in Sets:
            path = '/home/' + comp + '/catkin_ws/src/t42_control/rollout_t42/set/' + mode + Set + '/'
            files = glob.glob(path + mode + "*.txt")
            files_pkl = glob.glob(path + mode + "*.pkl")

            if len(files) == 0:
                continue
            for i in range(len(files)):
                action_file = files[i]
                if action_file.find('traj') > 0:
                    continue

############################# Later Uncomment ################################################################################################
                if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                    continue
############################# Later Uncomment ################################################################################################
                
                pklfile = action_file[:-3] + 'pkl'

                # To distribute rollout files between computers
                ja = pklfile.find('_goal')+5
                jb = ja + 1
                while not (pklfile[jb] == '_'):
                    jb += 1
                num = int(pklfile[ja:jb])
                
                jd = pklfile.find('run')+3         
                je = jd + 1
                while not (pklfile[je] == '_'):
                    je += 1
                run_idx = int(pklfile[jd:je])

                print('Rolling-out goal number ' + str(num) + ', run index ' + str(run_idx) + ': ' + action_file + '.')

                try:
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                except: 
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)
                    print(A.shape)

                Af = A.reshape((-1,))
                Pro = []
                for j in range(10):
                    print("Rollout number " + str(j) + ".")
                    #Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)
                    Sro = np.array(rollout_v2_srv(Af).states).reshape(-1,state_dim)                      
                    Pro.append(Sro)
                    with open(pklfile, 'wb') as f: 
                        pickle.dump(Pro, f)

    ############################# Evaluation ################################
    else:
        if sys.argv[0].find('src/')>=0:
            prefix=sys.argv[0][:sys.argv[0].find('src/')+4]
        else:
            prefix='../'
        rosparam_path=os.path.abspath(prefix+'param/settings.yaml')
        mac_cali_path=np.genfromtxt(rosparam_path,dtype=str)[1,1][1:-1]
        with open(mac_cali_path,'rb') as f:
            cali_info=np.array(pickle.load(f))
        rmat=cali_info[:9].reshape(3,3)
        tvec=cali_info[9:]

        init_mu=np.array([19.71323776,108.40877533,40.21877289,-65.48919678])

        for Set in Sets:
            C = np.array([[-35, 80],[-10, 100],[50, 100], [75, 80]])
            fig, ax = plt.subplots(figsize=(10,3.5))
            idx = [0,1,2,3]
            r=4.
            for i in idx:
                ctr = C[i]
                goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
                ax.add_artist(goal_plan)
                plt.text(ctr[0]-1.5, ctr[1]-1.5, str(i), fontsize=20)
            plt.plot(init_mu[0], init_mu[1], 'ok', markersize=16, color ='r',label='Start')
            plt.xlim([-50, 90])
            #plt.xlim([-60, 120])
            plt.ylim([70, 120])
            #plt.ylim([50, 120])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            results_path='/Users/zsbjltwjj/Downloads/t42_hand/t42_control/rollout_t42/set/'+ mode + Set +'/results/'
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            plt.savefig(results_path+'GOAL_LOCATIONS.png', dpi=200)


            # ==============================Statistics Evaluation=================
            evaluation=1

            def tracking_error(info, Straj, Pro_fail, Pro_suc):
                l=0
                for i in range(1,Straj.shape[0]):
                    l += np.linalg.norm(Straj[i,:2] - Straj[i-1,:2])
                if info=='suc':  
                    Sum_ls,l_ro_ls = [],[]
                    for ppp in Pro_suc:
                        Summ,l_ro = 0.,0.
                        pp,pp_valid_idx_ls=get_real_states(ppp)
                        Strajj=Straj[pp_valid_idx_ls]
                        for i in range(min(Straj.shape[0],pp.shape[0])):
                            Summ+=np.linalg.norm(Strajj[i,:2]-pp[i,:2])**2
                        Sum_ls.append(np.sqrt(Summ/min(Straj.shape[0],pp.shape[0])))
                        for i in range(1,pp.shape[0]):
                            l_ro+=np.linalg.norm(pp[i,:2] - pp[i-1,:2])
                        l_ro_ls.append(l_ro)

                    if len(Pro_fail)!=0:
                        l_ro_fail_ls=[]
                        for ppp in Pro_fail:
                            l_ro_fail=0
                            S_fail=get_real_states(ppp)[0]
                            for i in range(1,S_fail.shape[0]):
                                l_ro_fail += np.linalg.norm(S_fail[i,:2] - S_fail[i-1,:2])
                            l_ro_fail_ls.append(l_ro_fail)
                        return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls), np.mean(l_ro_fail_ls),np.std(l_ro_fail_ls)
                    else:
                        return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls), -1, -1
                else:
                    l_ro_fail_ls=[]
                    for ppp in Pro_fail:
                        l_ro_fail=0
                        S_fail=get_real_states(ppp)[0]
                        for i in range(1,S_fail.shape[0]):
                            l_ro_fail += np.linalg.norm(S_fail[i,:2] - S_fail[i-1,:2])                    
                        l_ro_fail_ls.append(l_ro_fail)
                    return -1,-1,l,-1,-1,np.mean(l_ro_fail_ls),np.std(l_ro_fail_ls)

            def calc_last_dist_to_goal(Pro_suc,Pro_fail,goal_loc):
                suc_last_dist,fail_last_dist=[],[]
                if len(Pro_suc)>0:
                    for ii in Pro_suc:
                        i=get_real_states(ii)[0]
                        suc_last_dist.append(np.linalg.norm(i[-1,:2]-goal_loc))
                    suc_last_dist_std=np.std(suc_last_dist)
                else:
                    suc_last_dist=[-1]
                    suc_last_dist_std=-1
                if len(Pro_fail):
                    for ii in Pro_fail:
                        i=get_real_states(ii)[0]
                        fail_last_dist.append(np.linalg.norm(i[-1,:2]-goal_loc))
                    fail_last_dist_std=np.std(fail_last_dist)
                else:
                    fail_last_dist=[-1]
                    fail_last_dist_std=-1
                return np.mean(suc_last_dist),suc_last_dist_std,np.mean(fail_last_dist),fail_last_dist_std

            Pastar = []
            Acastar = []
            if evaluation:
                new_C=[]
                for i in idx:
                    new_C.append(C[i])
                new_C=np.array(new_C)

                Sum = {mode: np.zeros((new_C.shape[0],17))} 

                
                planner = mode

                path = '/Users/zsbjltwjj/Downloads/t42_hand/t42_control/rollout_t42/set/' + mode + Set + '/'

                fo  = open(results_path + mode + '.txt', 'wt') 

                files = glob.glob(path + "*.pkl")

                for k in range(len(files)):

                    pklfile = files[k]
                    if pklfile.find('traj') > 0:
                        continue
                    if pklfile.find(mode) < 0:
                        continue
                    print("\nRunning pickle file: " + pklfile)
                    
                    ja = pklfile.find('_goal')+5
                    jb = ja + 1
                    while not (pklfile[jb] == '_'):
                        jb += 1
                    num = int(pklfile[ja:jb])
                    if num not in idx:
                        continue
                    goal_idx=idx.index(num)
                   
                    jd = pklfile.find('run')+3         
                    je = jd + 1
                    while not (pklfile[je] == '_'):
                        je += 1
                    run_idx = int(pklfile[jd:je])
                    try:
                        ctr = C[num, :] # Goal center
                    except:
                        raise ValueError('Goal Index Not Found!')

                    print('Goal Number ' + str(num) + ', with center:', ctr)

                    for j in range(len(pklfile)-1, 0, -1):
                        if pklfile[j] == '/':
                            break
                    file_name = pklfile[j+1:-4]

                    trajfile = pklfile[:-8] + 'traj.txt'
                    Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]
                    print('Plotting file number ' + str(k+1) + ': ' + file_name)
                    fig, ax = plt.subplots(figsize=(10,3.5))
                    goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
                    ax.add_artist(goal_plan)
                    plt.text(ctr[0]-1.5, ctr[1]-1.5, str(num), fontsize=20) 
                    
                    plt.plot(Straj[0,0], Straj[0,1], 'ok', markersize=16, color ='r',label='Start')
                    plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')
                    with open(pklfile,'rb') as f:  
                        Pro = pickle.load(f,encoding='latin')
                    A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]

                    for i in range(len(Pro)):
                        Pro[i]=get_transformed_states(Pro[i],rmat,tvec,with_med_filter,with_remove_d,remove_end_interval,filter_size)

                    plan_path_steps = Straj.shape[0]
                    
                    c = np.sum([(1 if x.shape[0]==plan_path_steps else 0) for x in Pro])
                    c = float(c) / len(Pro)*100
                    print("Finished episode success rate: " + str(c) + "%")

                    Pro_suc, Pro_fail = [],[]
                    for S in Pro:
                        if S.shape[0]==plan_path_steps:
                            Pro_suc.append(S)
                        else:
                            Pro_fail.append(S)
                    for kk in range(len(Pro)):
                        S = Pro[kk]
                        if S.shape[0] < plan_path_steps:
                            ii = S.shape[0]-1
                            while np.linalg.norm(S[ii,:2]-S[ii-1,:2]) > 4:
                                ii -= 1
                            Pro[kk] = S[:ii+1]
                    for kk in range(len(Pro_fail)):
                        S = Pro_fail[kk]
                        ii = S.shape[0]-1
                        while np.linalg.norm(S[ii,:2]-S[ii-1,:2]) > 4:
                            ii -= 1
                        Pro_fail[kk] = S[:ii+1]
                    if len(Pro_fail)!=0:
                        fail_path_steps = np.mean([i.shape[0] for i in Pro_fail])
                        fail_path_steps_std = np.std([i.shape[0] for i in Pro_fail])
                    else:
                        fail_path_steps = -1
                        fail_path_steps_std = -1
                    print("Planned path steps: "+str(plan_path_steps)+", failure path steps: "+str(fail_path_steps)+", failure path steps std: "+str(fail_path_steps_std))


                    p = 0
                    with_f_label,with_s_label=True,True
                    for SS in Pro:
                        S=get_real_states(SS)[0]
                        if np.linalg.norm(S[-1,:2]-ctr) > r:
                            print(np.linalg.norm(S[-1,:2]-ctr))
                            if with_f_label:
                                plt.plot(S[:,0], S[:,1], '-r',label='Failure rollout')
                                with_f_label=False
                            else:
                                plt.plot(S[:,0], S[:,1], '-r')
                            plt.plot(S[-1,0], S[-1,1], 'or')
                        else:
                            plt.plot(S[-1,0], S[-1,1], 'ob')
                            if with_s_label:
                                plt.plot(S[:,0], S[:,1], '-b',label='Success rollout')
                                with_s_label=False
                            else:
                                plt.plot(S[:,0], S[:,1], '-b')
                            p += 1
                    p = float(p) / len(Pro)*100
                    print("Reached goal success rate: " + str(p) + "%")


                    suc_last_dist,suc_last_dist_std,fail_last_dist,fail_last_dist_std=calc_last_dist_to_goal(Pro_suc,Pro_fail,ctr)
                    plan_last_dist=np.linalg.norm(Straj[-1,:2]-ctr)
                    print("Plan path last distace to goal: "+str(plan_last_dist)+"mm, success path last distance to goal: "+str(suc_last_dist)+" with std: "+str(suc_last_dist_std)+"mm, failure path last distance to goal: "+str(fail_last_dist)+" with std: "+str(fail_last_dist_std)+"mm.")


                    Smean = []
                    Sstd = []
                    if len(Pro_suc) > 0:
                        e, e_std, l, l_ro, l_ro_std, l_ro_fail, l_ro_fail_std = tracking_error('suc', Straj,Pro_fail,Pro_suc)
                    else:
                        e, e_std, l, l_ro, l_ro_std, l_ro_fail, l_ro_fail_std = tracking_error('no_suc', Straj,Pro_fail,Pro_suc)
                    print("Error: " + str(e)+" with std: "+str(e_std) + 'mm, plan path length: ' + str(l) + 'mm, success path length: ' + str(l_ro) +" with std: "+str(l_ro_std)+ 'mm, failure path length: ' + str(l_ro_fail) +" with std: "+str(l_ro_fail_std) +"mm.")
                    

                    loc=goal_idx
                    Sum[planner][loc, 0] = c # Success rate percent
                    Sum[planner][loc, 1] = p # Goal Reach rate percent
                    Sum[planner][loc, 2] = round(l, 2) # Planned path length
                    Sum[planner][loc, 3] = round(l_ro, 2) # Success path length
                    Sum[planner][loc, 4] = round(l_ro_std, 2) # Success path length std
                    Sum[planner][loc, 5] = round(l_ro_fail, 2) # Failure path length
                    Sum[planner][loc, 6] = round(l_ro_fail_std, 2) # Failure path length std
                    Sum[planner][loc, 7] = plan_path_steps # Plan path steps
                    Sum[planner][loc, 8] = fail_path_steps # Failure path steps
                    Sum[planner][loc, 9] = fail_path_steps_std # Failure path steps std
                    Sum[planner][loc, 10] = plan_last_dist # Plan path last distance to goal
                    Sum[planner][loc, 11] = suc_last_dist # Success path last distance to goal
                    Sum[planner][loc, 12] = suc_last_dist_std # Success path last distance to goal std
                    Sum[planner][loc, 13] = fail_last_dist # Failure path last distance to goal
                    Sum[planner][loc, 14] = fail_last_dist_std # Failure path last distance to goal std
                    Sum[planner][loc, 15] = round(e, 2) # Success path RMSE relative to plan path
                    Sum[planner][loc, 16] = round(e_std, 2) # Success path RMSE relative to plan path std
                    
                    if 1:
                        Pastar.append(p)
                        if e!=-1:
                            Acastar.append(e)

                    for i in range(len(pklfile)-1, 0, -1):
                        if pklfile[i] == '/':
                            break
                    fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
                    plt.xlim([-50, 90])
                    #plt.xlim([-60, 120])
                    plt.ylim([70, 120])
                    #plt.ylim([50, 120])
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.legend(prop={'size': 8})
                    plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png', dpi=200)

                fo.close()

                ################# Summary #################
                download_dir = results_path + '/summary.csv' 
                csv = open(download_dir, "w") 
                csv.write("Goal #,")
                for key in Sum.keys():
                    for _ in range(Sum[key].shape[1]):
                        csv.write(key + ',')
                csv.write('\n')
                csv.write(',,')
                for key in Sum.keys():
                    csv.write('success rate, goal reach rate, plan path length, success path length, success path length std, failure path length, failure path length std, plan path steps, failure path steps, failure path steps std, plan path last distance to goal, success path last distance to goal, success path last distance to goal std, failure path last distance to goal, failure path last distance to goal std, success path RMSE relative to plan path, success path RMSE relative to plan path std,')
                csv.write('\n')
                for loc in range(len(idx)):
                    goal=idx[loc]
                    csv.write(str(goal) + ',')
                    for key in Sum.keys():
                        for j in range(Sum[key].shape[1]):
                            if Sum[key][loc, j]==-1:
                                csv.write('-,')
                            else:
                                csv.write(str(Sum[key][loc, j]) + ',')
                    csv.write('\n')
                csv.write('Mean of all goals,')
                for key in Sum.keys():   
                    for j in range(Sum[key].shape[1]):
                        xx=np.delete(Sum[key][:, j],np.where(Sum[key][:, j]==-1))
                        if xx.shape[0]==0:
                            csv.write('-,')
                        else:
                            csv.write(str(np.mean(xx)) + ',')
                csv.write('\n')

                print(mode+": ")
                print("Mean goal reach rate: ", np.mean(np.array(Pastar)))
                if len(Acastar)!=0:
                    print("Mean error: ", np.mean(np.array(Acastar)))
                else:
                    print("Mean error: No Success Rollout")

