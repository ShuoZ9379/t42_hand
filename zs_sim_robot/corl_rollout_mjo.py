#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import gym
import glob,os,time,pickle
from scipy.io import loadmat
import sys

def get_y(plan_path):
    return -plan_path[:,0]-(plan_path[:,0]*plan_path[:,2]-plan_path[:,1]*plan_path[:,3])
def acro_calc_last_dist_to_goal(Pro,goal_height):
    ro_last_dist=[]
    for i in Pro:
        ro_last_dist.append(-min(0,get_y(i)[-1]-goal_height))
    ro_last_dist_std=np.std(ro_last_dist)
    return np.mean(ro_last_dist),ro_last_dist_std
def acro_tracking_error(Straj, Pro):
    l=0
    for i in range(1,Straj.shape[0]):
        l += np.linalg.norm(Straj[i] - Straj[i-1])
    Sum_ls,l_ro_ls = [],[]
    for pp in Pro:
        pp=get_y(pp)
        Summ,l_ro = 0.,0.
        for i in range(min(Straj.shape[0],pp.shape[0])):
            Summ+=(Straj[i]-pp[i])**2
        Sum_ls.append(np.sqrt(Summ/min(Straj.shape[0],pp.shape[0])))
        for i in range(1,pp.shape[0]):
            l_ro+=np.linalg.norm(pp[i] - pp[i-1])
        l_ro_ls.append(l_ro)
    return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls)

def reacher_calc_last_dist_to_goal(Pro,goal_loc):
    ro_last_dist=[]
    for i in Pro:
        ro_last_dist.append(np.linalg.norm(i[-1,6:8]-goal_loc))
    ro_last_dist_std=np.std(ro_last_dist)
    return np.mean(ro_last_dist),ro_last_dist_std
def reacher_tracking_error(Straj, Pro):
    l=0
    for i in range(1,Straj.shape[0]):
        l += np.linalg.norm(Straj[i,:2] - Straj[i-1,:2])
    print(l)
    Sum_ls,l_ro_ls = [],[]
    for pp in Pro:
        Summ,l_ro = 0.,0.
        for i in range(min(Straj.shape[0],pp.shape[0])):
            Summ+=np.linalg.norm(Straj[i,:2]-pp[i,6:8])**2
        Sum_ls.append(np.sqrt(Summ/min(Straj.shape[0],pp.shape[0])))
        for i in range(1,pp.shape[0]):
            l_ro+=np.linalg.norm(pp[i,6:8] - pp[i-1,6:8])
        l_ro_ls.append(l_ro)
    print(l_ro_ls)
    return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls)


def rollout_reacher_with_valid_range(plan_action_path,run_idx,num_ro):
    real_env=gym.make('Reacher-v2')
    real_env.seed(1000000+run_idx)
    obs=real_env.reset()
    goal_loc=obs[4:6]
    initial_state=obs[8:10]+obs[4:6]
    real_env.seed(1000000+run_idx)
    j=0
    Pro=[]
    while j <num_ro:
        obs=real_env.reset()
        if np.linalg.norm(obs[8:10]+obs[4:6]-initial_state)>0.0005:
            continue
        print("Rollout number " + str(j) + ".")
        ro_path=np.concatenate((obs[:4],obs[6:8],obs[8:10]+obs[4:6])).reshape(1,-1)
        for i in range(plan_action_path.shape[0]):
            obs=real_env.step(plan_action_path[i])[0]
            ro_path=np.concatenate((ro_path,np.concatenate((obs[:4],obs[6:8],obs[8:10]+obs[4:6])).reshape(1,-1)),axis=0)
        Pro.append(ro_path)
        j+=1
    return Pro

def rollout_acro(plan_action_path,run_idx,num_ro):
    real_env=gym.make('Acrobot-v1')
    real_env.seed(10000000+run_idx)
    obs=real_env.reset()
    initial_state=obs.copy()
    initial_y=-initial_state[0]-(initial_state[0]*initial_state[2]-initial_state[1]*initial_state[3])
    real_env.seed(10000000+run_idx)
    j=0
    Pro=[]
    while j <num_ro:
        obs=real_env.reset()
        #if np.linalg.norm(obs[8:10]+obs[4:6]-initial_state)>0.0005:
        #    continue
        print("Rollout number " + str(j) + ".")
        ro_path=obs.reshape(1,-1)
        for i in range(plan_action_path.shape[0]):
            obs,rwd,done,_=real_env.step(int(plan_action_path[i]))
            ro_path=np.concatenate((ro_path,obs.reshape(1,-1)),axis=0)
            if done:
                break
        Pro.append(ro_path)
        j+=1
    return Pro


rollout = int(sys.argv[1])
num_ro = 10
#ho_modes=['']
ho_modes=['_ho0.5','_ho0.6','_ho0.7','_ho0.8','_ho0.9','_ho0.95','_ho0.99']
ho_modes=['_ho0.99','_ho0.995','_ho0.999']
#Sets_ls = ['astar','policy']
Sets_ls = ['astar']
#env_id_ls = ['Reacher-v2','Acrobot-v1']
env_id_ls = ['Reacher-v2']

for ho_mode in ho_modes:
    for Sets in Sets_ls:
        for env_id in env_id_ls:
            set_modes_suf=False
            if Sets == 'policy':
                set_modes = ['']
            elif Sets == 'astar':
                if env_id == 'Acrobot-v1':
                    set_modes = ['quickest_search']
                elif env_id == 'Reacher-v2':
                    set_modes_suf=True
                    #set_modes = ['shortest_path','quickest_search']
                    set_modes = ['shortest_path']
            ############################# Rollout ################################
            if rollout:
                files_pkl = glob.glob('./mjo_'+Sets+'_eval_results'+ho_mode+'/'+env_id+'*.pkl')
                for set_mode in set_modes:
                    files = glob.glob('./mjo_'+Sets+'_eval_results'+ho_mode+'/'+env_id+'_'+set_mode+'*plan.txt')
                    if set_modes_suf:
                        res_folder='./mjo_'+Sets+'_eval_results'+ho_mode+'/'+env_id+'_'+set_mode+'_results/'
                    else:
                        res_folder='./mjo_'+Sets+'_eval_results'+ho_mode+'/'+env_id+'_results/'
                    if not os.path.exists(res_folder):
                        os.makedirs(res_folder)
                    for i in range(len(files)):
                        action_file = files[i]
                        if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                            continue
                        pklfile = action_file[:-3] + 'pkl'
                        ja = pklfile.find('_run')+4
                        jb = ja + 1
                        while not (pklfile[jb] == '_'):
                            jb += 1
                        num = int(pklfile[ja:jb])
                        print('Rolling-out goal number ' + str(num) + ': ' + action_file + '.')
                        if env_id=='Reacher-v2':
                            A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                            Pro = rollout_reacher_with_valid_range(plan_action_path=A,run_idx=num,num_ro=num_ro)
                        elif env_id=='Acrobot-v1':
                            A = np.loadtxt(action_file, delimiter=',', dtype=float)[:]
                            Pro = rollout_acro(plan_action_path=A,run_idx=num,num_ro=num_ro)
                        with open(pklfile, 'wb') as f: 
                            pickle.dump(Pro, f)

            ############################# Evaluation ################################
            else:

                if env_id=='Reacher-v2':
                    idx_ls=[1,2,5]
                    r=0.02
                    Sum = {set_mode: np.zeros((3,10)) for set_mode in set_modes} 

                    for set_mode in set_modes:
                        P_ls = []
                        Acc_ls = []
                        planner = set_mode
                        path = './mjo_'+Sets+'_eval_results'+ho_mode+'/'
                        files = glob.glob(path + env_id + '_' + set_mode + "*.pkl")
                        if set_modes_suf:
                            res_folder=path+env_id+'_'+set_mode+'_results/'
                        else:
                            res_folder=path+env_id+'_results/'

                        for k in range(len(files)):
                            pklfile = files[k]
                            print("\nRunning pickle file: " + pklfile)
                            ja = pklfile.find('_run')+4
                            jb = ja + 1
                            while not (pklfile[jb] == '_'):
                                jb += 1
                            num = int(pklfile[ja:jb])
                            goal_idx=idx_ls.index(num)
                            print('Goal Number ' + str(num) + '.')
                            env=gym.make('Reacher-v2')
                            env.seed(1000000+num)
                            goal_loc=env.reset()[4:6]
                            for j in range(len(pklfile)-1, 0, -1):
                                if pklfile[j] == '/':
                                    break
                            file_name = pklfile[j+1:-4]
                            trajfile = pklfile[:-8] + 'traj.txt'
                            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,6:8]


                            print('Plotting file number ' + str(k+1) + ': ' + file_name)
                            fig, ax = plt.subplots(figsize=(8,8))
                            plt.plot(Straj[0,0], Straj[0,1], 'ok', markersize=16, color ='r',label='Start')
                            goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), r, color='m')
                            plt.text(goal_loc[0]-0.01, goal_loc[1]-0.01,str(num),fontsize=20)
                            ax.add_artist(goal_plan)
                            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')

                            with open(pklfile,'rb') as f:  
                                Pro = pickle.load(f,encoding='latin')
                            plan_path_steps = Straj.shape[0]
                            print("Planned path steps: "+str(plan_path_steps)+".")

                            p = 0
                            with_f_label,with_s_label=True,True
                            for S in Pro:
                                if np.linalg.norm(S[-1,6:8]-goal_loc) > r:
                                    if with_f_label:
                                        plt.plot(S[:,6], S[:,7], '-r',label='Failure rollout')
                                        with_f_label=False
                                    else:
                                        plt.plot(S[:,6], S[:,7], '-r')
                                    plt.plot(S[-1,6], S[-1,7], 'or')
                                else:
                                    plt.plot(S[-1,6], S[-1,7], 'ob')
                                    if with_s_label:
                                        plt.plot(S[:,6], S[:,7], '-b', label='Success rollout')
                                        with_s_label=False
                                    else:
                                        plt.plot(S[:,6], S[:,7], '-b')
                                    p += 1
                            p = float(p) / len(Pro)*100
                            print("Reached goal success rate: " + str(p) + "%")
                        
                            ro_last_dist,ro_last_dist_std=reacher_calc_last_dist_to_goal(Pro,goal_loc)
                            plan_last_dist=np.linalg.norm(Straj[-1,:2]-goal_loc)
                            print("Plan path last distace to goal: "+str(plan_last_dist)+"mm, success path last distance to goal: "+str(ro_last_dist)+" with std: "+str(ro_last_dist_std)+"mm.")
                            
                            e, e_std, l, l_ro, l_ro_std = reacher_tracking_error(Straj,Pro)
                            print("Error: " + str(e)+" with std: "+str(e_std) + 'mm, plan path length: ' + str(l) + 'mm, success path length: ' + str(l_ro) +" with std: "+str(l_ro_std)+ 'mm.')
                            
                            Sum[planner][goal_idx, 0] = p # Goal Reach rate percent
                            Sum[planner][goal_idx, 1] = round(l, 5) # Planned path length
                            Sum[planner][goal_idx, 2] = round(l_ro, 5) # Rollout path length
                            Sum[planner][goal_idx, 3] = round(l_ro_std, 5) # Rollout path length std
                            Sum[planner][goal_idx, 4] = plan_path_steps # Plan path steps
                            Sum[planner][goal_idx, 5] = round(plan_last_dist,5) # Plan path last distance to goal
                            Sum[planner][goal_idx, 6] = round(ro_last_dist,5) # Rollout path last distance to goal
                            Sum[planner][goal_idx, 7] = round(ro_last_dist_std,5) # Rollout path last distance to goal std
                            Sum[planner][goal_idx, 8] = round(e, 5) # Rollout path RMSE relative to plan path
                            Sum[planner][goal_idx, 9] = round(e_std, 5) # Rollout path RMSE relative to plan path std

                            P_ls.append(p)
                            Acc_ls.append(e)
        
                            plt.xlim([-0.22, 0.22])
                            plt.ylim([-0.22, 0.22])
                            plt.xlabel('x')
                            plt.ylabel('y')
                            plt.legend()
                            plt.savefig(res_folder + file_name + '.png', dpi=200)
             
                        ################# Summary #################
                        download_dir = res_folder +'summary.csv' 
                        csv = open(download_dir, "w") 
                        csv.write("Goal #,")
                        for _ in range(Sum[set_mode].shape[1]):
                            csv.write(set_mode + ',')
                        csv.write('\n')
                        csv.write(',')
                        csv.write('goal reach rate, plan path length, rollout path length, rollout path length std, plan path steps, plan path last distance to goal, rollout path last distance to goal, rollout path last distance to goal std, rollout path RMSE relative to plan path, rollout path RMSE relative to plan path std,')
                        csv.write('\n')
                        for loc in range(len(idx_ls)):
                            goal=idx_ls[loc]
                            csv.write(str(goal) + ',')
                            for j in range(Sum[set_mode].shape[1]):
                                if Sum[set_mode][loc, j]==-1:
                                    csv.write('-,')
                                else:
                                    csv.write(str(Sum[set_mode][loc, j]) + ',')
                            csv.write('\n')
                        csv.write('Mean of all goals,')   
                        for j in range(Sum[set_mode].shape[1]):
                            csv.write(str(np.mean(Sum[set_mode][:, j])) + ',')
                        csv.write('\n')

                        print(set_mode+": ")
                        print("Mean goal reach rate: ", np.mean(np.array(P_ls)))
                        print("Mean error: ", np.mean(np.array(Acc_ls)))


                elif env_id=='Acrobot-v1':
                    idx_ls=[1]
                    Sum = {set_mode: np.zeros((1,12)) for set_mode in set_modes} 

                    for set_mode in set_modes:
                        P_ls = []
                        Acc_ls = []
                        planner = set_mode
                        path = './mjo_'+Sets+'_eval_results/'
                        files = glob.glob(path + env_id + '_' + set_mode + "*.pkl")
                        if set_modes_suf:
                            res_folder=path+env_id+'_'+set_mode+'_results/'
                        else:
                            res_folder=path+env_id+'_results/'

                        for k in range(len(files)):
                            pklfile = files[k]
                            print("\nRunning pickle file: " + pklfile)
                            ja = pklfile.find('_run')+4
                            jb = ja + 1
                            while not (pklfile[jb] == '_'):
                                jb += 1
                            num = int(pklfile[ja:jb])
                            goal_idx=idx_ls.index(num)
                            print('Goal Number ' + str(num) + '.')
                            goal_height=1.0
                            for j in range(len(pklfile)-1, 0, -1):
                                if pklfile[j] == '/':
                                    break
                            file_name = pklfile[j+1:-4]
                            trajfile = pklfile[:-8] + 'traj.txt'
                            plan_path = np.loadtxt(trajfile, delimiter=',', dtype=float)
                            Straj=get_y(plan_path)


                            print('Plotting file number ' + str(k+1) + ': ' + file_name)
                            fig, ax = plt.subplots(figsize=(8,8))
                            plt.plot([goal_height for sss in range(Straj.shape[0])],'-m')
                            plt.plot(Straj, '-k', linewidth = 2.7, label='Planned Y path')

                            with open(pklfile,'rb') as f:  
                                Pro = pickle.load(f,encoding='latin')
                            plan_path_steps = Straj.shape[0]
                            ro_path_steps_ls=[]
                            for S in Pro:
                                ro_path_steps_ls.append(S.shape[0])
                            ro_path_steps,ro_path_steps_std=np.mean(ro_path_steps_ls),np.std(ro_path_steps_ls)
                            print("Planned path steps: "+str(plan_path_steps)+", rollout path steps: "+str(ro_path_steps)+" with std: "+str(ro_path_steps_std)+".")

                            p = 0
                            with_f_label,with_s_label=True,True
                            for S in Pro:
                                S_y=get_y(S)
                                if S_y[-1]<goal_height:
                                    if with_f_label:
                                        plt.plot(S_y, '-r',label='Failure Y path')
                                        with_f_label=False
                                    else:
                                        plt.plot(S_y, '-r')
                                    plt.plot(S_y.shape[0]-1,S_y[-1], 'or')
                                else:
                                    plt.plot(S_y.shape[0]-1,S_y[-1], 'ob')
                                    if with_s_label:
                                        plt.plot(S_y, '-b',label='Success Y path')
                                        with_s_label=False
                                    else:
                                        plt.plot(S_y, '-b')
                                    p += 1
                            p = float(p) / len(Pro)*100
                            print("Reached goal success rate: " + str(p) + "%")
                        
                            ro_last_dist,ro_last_dist_std=acro_calc_last_dist_to_goal(Pro,goal_height)
                            plan_last_dist=-min(Straj[-1]-goal_height,0)
                            print("Plan path last distace to goal: "+str(plan_last_dist)+"mm, success path last distance to goal: "+str(ro_last_dist)+" with std: "+str(ro_last_dist_std)+"mm.")
                            
                            e, e_std, l, l_ro, l_ro_std = acro_tracking_error(Straj,Pro)
                            print("Error: " + str(e)+" with std: "+str(e_std) + 'mm, plan path length: ' + str(l) + 'mm, success path length: ' + str(l_ro) +" with std: "+str(l_ro_std)+ "mm.")
                            
                            Sum[planner][goal_idx, 0] = p # Goal Reach rate percent
                            Sum[planner][goal_idx, 1] = round(l, 5) # Planned path length
                            Sum[planner][goal_idx, 2] = round(l_ro, 5) # Rollout path length
                            Sum[planner][goal_idx, 3] = round(l_ro_std, 5) # Rollout path length std
                            Sum[planner][goal_idx, 4] = plan_path_steps # Plan path steps
                            Sum[planner][goal_idx, 5] = ro_path_steps # Rollout path steps
                            Sum[planner][goal_idx, 6] = round(ro_path_steps_std,2) # Rollout path steps std
                            Sum[planner][goal_idx, 7] = round(plan_last_dist,5) # Plan path last distance to goal
                            Sum[planner][goal_idx, 8] = round(ro_last_dist,5) # Rollout path last distance to goal
                            Sum[planner][goal_idx, 9] = round(ro_last_dist_std,5) # Rollout path last distance to goal std
                            Sum[planner][goal_idx, 10] = round(e, 5) # Rollout path RMSE relative to plan path
                            Sum[planner][goal_idx, 11] = round(e_std, 5) # Rollout path RMSE relative to plan path std

                            P_ls.append(p)
                            Acc_ls.append(e)
        
                            plt.ylim([-2.2, 2.2])
                            plt.xlabel('Steps')
                            plt.ylabel('Y position')
                            plt.legend(prop={'size': 8})
                            plt.savefig(res_folder + file_name + '.png', dpi=200)
             
                        ################# Summary #################
                        download_dir = res_folder +'summary.csv' 
                        csv = open(download_dir, "w") 
                        csv.write("Goal #,")
                        for _ in range(Sum[set_mode].shape[1]):
                            csv.write(set_mode + ',')
                        csv.write('\n')
                        csv.write(',')
                        csv.write('goal reach rate, plan path length, rollout path length, rollout path length std, plan path steps, rollout path steps, rollout path steps std, plan path last distance to goal, rollout path last distance to goal, rollout path last distance to goal std, rollout path RMSE relative to plan path, rollout path RMSE relative to plan path std,')
                        csv.write('\n')
                        for loc in range(len(idx_ls)):
                            goal=idx_ls[loc]
                            csv.write(str(goal) + ',')
                            for j in range(Sum[set_mode].shape[1]):
                                if Sum[set_mode][loc, j]==-1:
                                    csv.write('-,')
                                else:
                                    csv.write(str(Sum[set_mode][loc, j]) + ',')
                            csv.write('\n')
                        csv.write('Mean of all goals,')   
                        for j in range(Sum[set_mode].shape[1]):
                            csv.write(str(np.mean(Sum[set_mode][:, j])) + ',')
                        csv.write('\n')

                        print(set_mode+": ")
                        print("Mean goal reach rate: ", np.mean(np.array(P_ls)))
                        print("Mean error: ", np.mean(np.array(Acc_ls)))
        

