import pickle
import numpy as np
color='red'
obj='cyl30'
data_type='d'
train_idx='9'
base_path='/Users/zsbjltwjj/Desktop/'

del_path=base_path+color+'_data/'+'zs_raw_train_'+obj+'_'+color+'_'+data_type+'_v'+train_idx+'.obj'
with open(del_path,'rb') as filehandler:
    memory=pickle.load(filehandler,encoding='latin1')
print(len(memory))
ls=[[1898,2359],[3508,4478],[6262,7129],[42798,43254],[106420,106742],[136302,137933]]
num=0
for slc in ls:
    del memory[slc[0]-num:slc[1]-num]
    num+=slc[1]-slc[0]
with open(del_path,'wb') as f:
    pickle.dump(memory,f)
print(len(memory))
