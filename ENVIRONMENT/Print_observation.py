import numpy 
import torch
from socnavenv import SocNavEnv
env = SocNavEnv()
#with numpy.load('./dataset/Test_dataset.npz') as data:
with numpy.load('./dataset/Train_dataset.npz') as data:
    
    a = data['observations.npy']
    #print(a)
x = numpy.array(a)
#x = torch.from_numpy(x)
#print(x.shape)
#print( len(x))

mu_dataset = []

for i in range(1):#len(x)):
    mu_dataset.append(x[i]) 


mu_dataset = numpy.array(mu_dataset)


print(mu_dataset.shape)
print(mu_dataset)