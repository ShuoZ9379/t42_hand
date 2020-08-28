#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import gym
import glob,os,time,pickle
from scipy.io import loadmat
import sys


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
    #print(l)
    Sum_ls,l_ro_ls = [],[]
    for pp in Pro:
        Summ,l_ro = 0.,0.
        for i in range(min(Straj.shape[0],pp.shape[0])):
            Summ+=np.linalg.norm(Straj[i,:2]-pp[i,6:8])**2
        Sum_ls.append(np.sqrt(Summ/min(Straj.shape[0],pp.shape[0])))
        for i in range(1,pp.shape[0]):
            l_ro+=np.linalg.norm(pp[i,6:8] - pp[i-1,6:8])
        l_ro_ls.append(l_ro)
    #print(l_ro_ls)
    return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls)


def lqr_rollout_reacher_with_valid_range(K_ls,plan_traj_path,plan_action_path,run_idx,num_ro,big_goal_radius=0.02):
    real_env=gym.make('Reacher-v2')
    real_env.seed(1000000+run_idx)
    obs=real_env.reset()
    goal_loc=obs[4:6]
    initial_state=obs[8:10]+obs[4:6]
    real_env.seed(1000000+run_idx)
    j=0
    Tro,Aro = [],[]
    while j <num_ro:
        obs=real_env.reset()
        if np.linalg.norm(obs[8:10]+obs[4:6]-initial_state)>0.0005:
            continue
        print("Rollout number " + str(j) + ".")
        tro=np.concatenate((obs[:4],obs[6:8],obs[8:10]+obs[4:6])).reshape(1,-1)
        aro=np.zeros((0,plan_action_path.shape[1]))
        for step in range(plan_action_path.shape[0]):
            new_a=K_ls[step].dot(tro[-1,:]-plan_traj_path[step,:])+plan_action_path[step,:]
            res=real_env.step(new_a)
            new_t=np.concatenate((res[0][:4],res[0][6:8],res[0][8:10]+res[0][4:6])).reshape(1,plan_traj_path.shape[1])
            aro = np.concatenate((aro,new_a.reshape((1,-1))))
            tro = np.concatenate((tro,new_t)) 
            #print(np.linalg.norm(tro[-1,-2:]-goal_loc))
            if np.linalg.norm(tro[-1,-2:]-goal_loc) <= big_goal_radius:
                print('[LQR] Goal Reach')
                break
        Tro.append(tro)
        Aro.append(aro)
        j+=1
    return Tro,Aro

rollout = int(sys.argv[1])
num_ro = 10
big_goal_radius=0.02
ho_modes=['_ho0.999']
Sets_ls = ['astar']
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
                for set_mode in set_modes:

                    lqr_ro_path='./lqr_mjo_' + Sets + '_eval_results'+ho_mode + '/'
                    if not os.path.exists(lqr_ro_path):
                        os.makedirs(lqr_ro_path)

                    if set_modes_suf:
                        res_folder=lqr_ro_path+env_id+'_'+set_mode+'_results/'
                    else:
                        res_folder=lqr_ro_path+env_id+'_results/'
                    if not os.path.exists(res_folder):
                        os.makedirs(res_folder)

                    path = './mjo_' + Sets + '_eval_results'+ho_mode + '/'

                    for goal_idx in [1,2,5]:
                        k_file = './lqr_k/'+Sets+ho_mode+'_'+env_id+'_goal'+str(goal_idx)+'_K'
                        with open(k_file,'rb') as f:
                            K_ls=pickle.load(f)
                        traj_file = glob.glob(path + env_id+'_'+set_mode +"*_run"+str(goal_idx)+ "_traj.txt")[0]
                        action_file = glob.glob(path + env_id+'_'+set_mode +"*_run"+str(goal_idx)+ "_plan.txt")[0]
                        files_pkl = glob.glob(lqr_ro_path + env_id+'_'+set_mode + "*.pkl")


    ############################# Later Uncomment ################################################################################################
                        #if any(action_file[len(path):-3] + 'pkl' in f for f in files_pkl):
                        #    continue
    ############################# Later Uncomment ################################################################################################
                        
                        action_pklfile = lqr_ro_path+action_file[len(path):-3] + 'pkl'
                        traj_pklfile = lqr_ro_path+traj_file[len(path):-3] + 'pkl'
                        try:
                            A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                            T = np.loadtxt(traj_file, delimiter=',', dtype=float)[:,:8]
                        except: 
                            A = np.loadtxt(action_file, delimiter=',', dtype=float)
                            T = np.loadtxt(traj_file, delimiter=',', dtype=float)
  
                        print('Rolling-out goal number ' + str(goal_idx) + '.')
                        if env_id=='Reacher-v2':
                            Tro,Aro = lqr_rollout_reacher_with_valid_range(K_ls,T,A,goal_idx,num_ro,big_goal_radius)
                        with open(action_pklfile, 'wb') as f: 
                                pickle.dump(Aro, f)
                        with open(traj_pklfile, 'wb') as f: 
                                pickle.dump(Tro, f)

            ############################# Evaluation ################################
            else:
                if env_id=='Reacher-v2':
                    idx_ls=[1,2,5]
                    r=0.02
                    Sum = {set_mode: np.zeros((len(idx_ls),12)) for set_mode in set_modes} 

                    for set_mode in set_modes:
                        P_ls = []
                        Acc_ls = []
                        planner = set_mode
                        lqr_ro_path='./lqr_mjo_' + Sets + '_eval_results'+ho_mode + '/'
                        path = './mjo_' + Sets + '_eval_results'+ho_mode + '/'

                        if set_modes_suf:
                            res_folder=lqr_ro_path+env_id+'_'+set_mode+'_results/'
                        else:
                            res_folder=lqr_ro_path+env_id+'_results/'

                        for k in range(len(idx_ls)):
                            num=idx_ls[k]
                            trajfile = glob.glob(path + env_id+'_'+set_mode +'*_run'+str(num)+'_traj.txt')[0]
                            #action_file = glob.glob(path + env_id+'_'+set_mode +'*_run'+str(num)+'_plan.txt')[0]
                            #action_pklfile = lqr_ro_path+action_file[len(path):-3] + 'pkl'
                            traj_pklfile = lqr_ro_path+trajfile[len(path):-3] + 'pkl'
                            if traj_pklfile.find(set_mode) < 0:
                                continue
                            print("\nRunning pickle file: " + traj_pklfile)
                            goal_idx=idx_ls.index(num)
                            print('Goal Number ' + str(num) + '.')

                            env=gym.make('Reacher-v2')
                            env.seed(1000000+num)
                            goal_loc=env.reset()[4:6]
                            for j in range(len(traj_pklfile)-1, 0, -1):
                                if traj_pklfile[j] == '/':
                                    break
                            file_name = traj_pklfile[j+1:-4]
                            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,6:8]


                            print('Plotting file number ' + str(k+1) + ': ' + file_name)
                            fig, ax = plt.subplots(figsize=(8,8))
                            plt.plot(Straj[0,0], Straj[0,1], 'ok', markersize=16, color ='r',label='Start')
                            goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), r, color='m')
                            plt.text(goal_loc[0]-0.01, goal_loc[1]-0.01,str(num),fontsize=20)
                            ax.add_artist(goal_plan)
                            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')

                            with open(traj_pklfile,'rb') as f:  
                                Pro = pickle.load(f,encoding='latin')
                            plan_path_steps = Straj.shape[0]
                            ro_path_steps = np.mean([i.shape[0] for i in Pro])
                            ro_path_steps_std = np.std([i.shape[0] for i in Pro])
                            print("Planned path steps: "+str(plan_path_steps)+", lqr ro path steps: "+str(ro_path_steps)+", with std: "+str(ro_path_steps_std))

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
                            print("Plan path last distace to goal: "+str(plan_last_dist)+"mm, lqr path last distance to goal: "+str(ro_last_dist)+" with std: "+str(ro_last_dist_std)+"mm.")
                            
                            e, e_std, l, l_ro, l_ro_std = reacher_tracking_error(Straj,Pro)
                            print("Error: " + str(e)+" with std: "+str(e_std) + 'mm, plan path length: ' + str(l) + 'mm, lqr path length: ' + str(l_ro) +" with std: "+str(l_ro_std)+ 'mm.')
                            
                            Sum[planner][goal_idx, 0] = p # Goal Reach rate percent
                            Sum[planner][goal_idx, 1] = round(l, 5) # Planned path length
                            Sum[planner][goal_idx, 2] = round(l_ro, 5) # Rollout path length
                            Sum[planner][goal_idx, 3] = round(l_ro_std, 5) # Rollout path length std
                            Sum[planner][goal_idx, 4] = plan_path_steps # Plan path steps
                            Sum[planner][goal_idx, 5] = ro_path_steps # lqr ro path steps
                            Sum[planner][goal_idx, 6] = ro_path_steps_std # lqr ro path steps std
                            Sum[planner][goal_idx, 7] = round(plan_last_dist,5) # Plan path last distance to goal
                            Sum[planner][goal_idx, 8] = round(ro_last_dist,5) # Rollout path last distance to goal
                            Sum[planner][goal_idx, 9] = round(ro_last_dist_std,5) # Rollout path last distance to goal std
                            Sum[planner][goal_idx, 10] = round(e, 5) # Rollout path RMSE relative to plan path
                            Sum[planner][goal_idx, 11] = round(e_std, 5) # Rollout path RMSE relative to plan path std

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
                            csv.write('lqr_'+set_mode + ',')
                        csv.write('\n')
                        csv.write(',')
                        csv.write('goal reach rate, plan path length, lqr ro path length, lqr ro path length std, plan path steps, lqr ro path steps, lqr ro path steps std, plan path last distance to goal, lqr ro path last distance to goal, lqr ro path last distance to goal std, lqr ro path RMSE relative to plan path, lqr ro path RMSE relative to plan path std,')
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

                        print('lqr_'+set_mode+": ")
                        print("Mean goal reach rate: ", np.mean(np.array(P_ls)))
                        print("Mean error: ", np.mean(np.array(Acc_ls)))