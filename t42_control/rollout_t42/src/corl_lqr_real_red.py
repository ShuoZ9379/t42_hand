#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
import time,copy
import glob,os,sys
#rosrun path/to/corl_lqr_real_red.py
# If wanna stop as soon as not res.grasped, then delete z axix paral condition and 5 markers(except obj marker) detection condition + keep record of object marker tracking anyway.
# If wanna post_process traj to make it more smooth or beatiful, then refer to funcs in corl_rollout_real_red.py

rollout = 1
comp = 'szhang'
state_dim = 4
action_dim = 2

mode_ls=['astar']
ah_with_goal_loc=1
big_goal_radius=4
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
        from rollout_t42.srv import reset,StepOnlineReq

        reset_srv=rospy.ServiceProxy('/rollout/ResetOnline', reset)
        next_step_srv=rospy.ServiceProxy('/rollout/StepOnlineOneStep', StepOnlineReq)
        rospy.init_node('run_lqr_set', anonymous=True)

        ho_ls=['']
        for ho_suf in ho_ls:
            for Set in Sets:
                lqr_ro_path='/home/' + comp + '/catkin_ws/src/t42_control/rollout_t42/set/lqr_' +mode+ Set + ho_suf + '/'
                if not os.path.exists(lqr_ro_path):
                    os.makedirs(lqr_ro_path)
                path = '/home/' + comp + '/catkin_ws/src/t42_control/rollout_t42/set/' + mode + Set + ho_suf+'/'
                for goal_idx in [0,1,2,3]:
                    k_file = '/home/' + comp + '/catkin_ws/src/zs_sim_robot/lqr_k/'+mode+ho_suf+'_real_ah_goal'+str(goal_idx)+'_K'
                    with open(k_file,'rb') as f:
                        K_ls=pickle.load(f)
                    traj_file = glob.glob(path + mode +"_goal"+str(goal_idx)+ "*_traj.txt")[0]
                    action_file = glob.glob(path + mode +"_goal"+str(goal_idx)+ "*_plan.txt")[0]
                    files_pkl = glob.glob(lqr_ro_path + mode + "*.pkl")

        ############################# Later Uncomment ################################################################################################
                    #if any(action_file[len(path):-3] + 'pkl' in f for f in files_pkl):
                    #    continue
        ############################# Later Uncomment ################################################################################################

                    action_pklfile = lqr_ro_path+action_file[len(path):-3] + 'pkl'
                    traj_pklfile = lqr_ro_path+traj_file[len(path):-3] + 'pkl'

                    # To distribute rollout files between computers
                    jd = action_pklfile.find('run')+3         
                    je = jd + 1
                    while not (action_pklfile[je] == '_'):
                        je += 1
                    run_idx = int(action_pklfile[jd:je])

                    print('LQR closed loop control for goal number ' + str(goal_idx) + ', run index ' + str(run_idx)+'.')

                    try:
                        A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                        T = np.loadtxt(traj_file, delimiter=',', dtype=float)[:,:4]
                    except: 
                        A = np.loadtxt(action_file, delimiter=',', dtype=float)
                        T = np.loadtxt(traj_file, delimiter=',', dtype=float)

                    Tro,Aro = [],[]
                    for j in range(10):
                        print("Rollout number " + str(j) + ".")
                        tro=np.array(reset_srv(goal_idx,big_goal_radius).states).reshape(1,state_dim)
                        init_tro=copy.copy(tro.reshape((-1,)))
                        aro=np.zeros((0,action_dim))
                        for step in range(A.shape[0]):
                            new_a=K_ls[step].dot(np.concatenate((tro[-1,:],init_tro))-np.concatenate((T[step,:],T[0,:])))+A[step,:]
                            res=next_step_srv(new_a)
                            new_t=np.array(res.states).reshape(1,state_dim)
                            aro = np.concatenate((aro,new_a.reshape((-1,action_dim))))
                            tro = np.concatenate((tro,new_t))  
                            if not res.success or res.goal_reach:
                                break
                            ###might be changed: e.g, count fail >11;
                            elif not res.grasped:
                                break
                        Tro.append(tro)
                        Aro.append(aro)
                        with open(action_pklfile, 'wb') as f: 
                            pickle.dump(Aro, f)
                        with open(traj_pklfile, 'wb') as f: 
                            pickle.dump(Tro, f)

    ############################# Evaluation ################################
    else:
        ho_ls=['']
        for ho_suf in ho_ls:
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
                results_path='/Users/zsbjltwjj/Downloads/t42_hand/t42_control/rollout_t42/set/lqr_'+ mode + Set +ho_suf+'/results/'
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
                        for pp in Pro_suc:
                            Summ,l_ro = 0.,0.
                            for i in range(min(Straj.shape[0],pp.shape[0])):
                                Summ+=np.linalg.norm(Straj[i,:2]-pp[i,:2])**2
                            Sum_ls.append(np.sqrt(Summ/min(Straj.shape[0],pp.shape[0])))
                            for i in range(1,pp.shape[0]):
                                l_ro+=np.linalg.norm(pp[i,:2] - pp[i-1,:2])
                            l_ro_ls.append(l_ro)
                        return np.mean(Sum_ls), np.std(Sum_ls), l, np.mean(l_ro_ls), np.std(l_ro_ls), -1, -1

                def calc_last_dist_to_goal(Pro_suc,Pro_fail,goal_loc):
                    suc_last_dist,fail_last_dist=[],[]
                    for i in Pro_suc:
                        suc_last_dist.append(np.linalg.norm(i[-1,:2]-goal_loc))
                    suc_last_dist_std=np.std(suc_last_dist)
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

                    Sum = {mode: np.zeros((new_C.shape[0],12))} 

                    
                    planner = mode
                    lqr_ro_path='/Users/zsbjltwjj/Downloads/t42_hand/t42_control/rollout_t42/set/lqr_' +mode+ Set + ho_suf + '/'
                    path = '/Users/zsbjltwjj/Downloads/t42_hand/t42_control/rollout_t42/set/' +mode+ Set + ho_suf + '/'

                    fo  = open(results_path+ 'lqr_'  + mode + '.txt', 'wt') 


                    for k in range(len(idx)):
                        num=idx[k]
                        k_file = '/Users/zsbjltwjj/Downloads/t42_hand/zs_sim_robot/lqr_k/'+mode+ho_suf+'_real_ah_goal'+str(num)+'_K'
                        with open(k_file,'rb') as f:
                            K_ls=pickle.load(f)
                        trajfile = glob.glob(path + mode +"_goal"+str(num)+ "*_traj.txt")[0]
                        #action_file = glob.glob(path + mode +"_goal"+str(goal_idx)+ "*_plan.txt")[0]
                        #action_pklfile = lqr_ro_path+action_file[len(path):-3] + 'pkl'
                        traj_pklfile = lqr_ro_path+trajfile[len(path):-3] + 'pkl'
                        if traj_pklfile.find(mode) < 0:
                            continue
                        print("\nRunning pickle file: " + traj_pklfile)
                        goal_idx=idx.index(num)
                       
                        jd = traj_pklfile.find('run')+3         
                        je = jd + 1
                        while not (traj_pklfile[je] == '_'):
                            je += 1
                        run_idx = int(traj_pklfile[jd:je])
                        try:
                            ctr = C[num, :] # Goal center
                        except:
                            raise ValueError('Goal Index Not Found!')

                        print('Goal Number ' + str(num) + ', with center:', ctr)

                        for j in range(len(traj_pklfile)-1, 0, -1):
                            if traj_pklfile[j] == '/':
                                break
                        file_name = traj_pklfile[j+1:-4]

                        Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]
                        print('Plotting file number ' + str(k+1) + ': ' + file_name)
                        fig, ax = plt.subplots(figsize=(10,3.5))
                        goal_plan = plt.Circle((ctr[0], ctr[1]), r, color='m')
                        ax.add_artist(goal_plan)
                        plt.text(ctr[0]-1.5, ctr[1]-1.5, str(num), fontsize=20) 
                        
                        plt.plot(Straj[0,0], Straj[0,1], 'ok', markersize=16, color ='r',label='Start')
                        plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')
                        with open(traj_pklfile,'rb') as f:  
                            Pro = pickle.load(f,encoding='latin')
                
                        plan_path_steps = Straj.shape[0]
                        
                        Pro_suc, Pro_fail = [],[]
                        for S in Pro:
                            Pro_suc.append(S)
                        ro_path_steps = np.mean([i.shape[0] for i in Pro_suc])
                        ro_path_steps_std = np.std([i.shape[0] for i in Pro_suc])
                        print("Planned path steps: "+str(plan_path_steps)+", lqr ro path steps: "+str(ro_path_steps)+", with std: "+str(ro_path_steps_std))


                        p = 0
                        with_f_label,with_s_label=True,True
                        for S in Pro:
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


                        ro_last_dist,ro_last_dist_std,fail_last_dist,fail_last_dist_std=calc_last_dist_to_goal(Pro_suc,Pro_fail,ctr)
                        plan_last_dist=np.linalg.norm(Straj[-1,:2]-ctr)
                        print("Plan path last distace to goal: "+str(plan_last_dist)+"mm, lqr ro path last distance to goal: "+str(ro_last_dist)+" with std: "+str(ro_last_dist_std)+"mm.")


                        Smean = []
                        Sstd = []
                    
                        e, e_std, l, l_ro, l_ro_std, l_ro_fail, l_ro_fail_std = tracking_error('suc', Straj,Pro_fail,Pro_suc)
                        print("Error: " + str(e)+" with std: "+str(e_std) + 'mm, plan path length: ' + str(l) + 'mm, lqr ro path length: ' + str(l_ro) +" with std: "+str(l_ro_std)+ 'mm.')

                        loc=goal_idx
                        Sum[planner][loc, 0] = p # Goal Reach rate percent
                        Sum[planner][loc, 1] = round(l, 2) # Planned path length
                        Sum[planner][loc, 2] = round(l_ro, 2) # lqr ro path length
                        Sum[planner][loc, 3] = round(l_ro_std, 2) # lqr ro path length std
                        #Sum[planner][loc, 5] = round(l_ro_fail, 2) # Failure path length
                        #Sum[planner][loc, 6] = round(l_ro_fail_std, 2) # Failure path length std
                        Sum[planner][loc, 4] = plan_path_steps # Plan path steps
                        Sum[planner][loc, 5] = ro_path_steps # lqr ro path steps
                        Sum[planner][loc, 6] = ro_path_steps_std # lqr ro path steps std
                        Sum[planner][loc, 7] = plan_last_dist # Plan path last distance to goal
                        Sum[planner][loc, 8] = ro_last_dist # lqr ro path last distance to goal
                        Sum[planner][loc, 9] = ro_last_dist_std # lqr ro path last distance to goal std
                        #Sum[planner][loc, 13] = fail_last_dist # Failure path last distance to goal
                        #Sum[planner][loc, 14] = fail_last_dist_std # Failure path last distance to goal std
                        Sum[planner][loc, 10] = round(e, 2) # Success path RMSE relative to plan path
                        Sum[planner][loc, 11] = round(e_std, 2) # Success path RMSE relative to plan path std
                        
                        if 1:
                            Pastar.append(p)
                            if e!=-1:
                                Acastar.append(e)

                        for i in range(len(traj_pklfile)-1, 0, -1):
                            if traj_pklfile[i] == '/':
                                break
                        fo.write('lqr_'+traj_pklfile[i+1:-4] + ': ' + str(p) + '\n')
                        plt.xlim([-50, 90])
                        #plt.xlim([-60, 120])
                        plt.ylim([70, 120])
                        #plt.ylim([50, 120])
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.legend(prop={'size': 8})
                        plt.savefig(results_path + '/lqr_' + traj_pklfile[i+1:-4] + '.png', dpi=200)

                    fo.close()

                    ################# Summary #################
                    download_dir = results_path + '/summary.csv' 
                    csv = open(download_dir, "w") 
                    csv.write("Goal #,")
                    for key in Sum.keys():
                        for _ in range(Sum[key].shape[1]):
                            csv.write('lqr_'+key + ',')
                    csv.write('\n')
                    csv.write(',,')
                    for key in Sum.keys():
                        csv.write('goal reach rate, plan path length, lqr ro path length, lqr ro path length std, plan path steps, lqr ro path steps, lqr ro path steps std, plan path last distance to goal, lqr ro path last distance to goal, lqr ro path last distance to goal std, lqr ro path RMSE relative to plan path, lqr ro path RMSE relative to plan path std,')
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

                    print('lqr_'+mode+": ")
                    print("Mean goal reach rate: ", np.mean(np.array(Pastar)))
                    if len(Acastar)!=0:
                        print("Mean error: ", np.mean(np.array(Acastar)))

