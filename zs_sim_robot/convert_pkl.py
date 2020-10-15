import pickle,sys
name=sys.argv[1]
#print(name)
with open (name,'rb') as f:
    your_object=pickle.load(f)
with open (name,'wb') as f:
    pickle.dump(your_object, f, protocol=2)
