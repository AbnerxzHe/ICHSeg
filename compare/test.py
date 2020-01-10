import torch
import matplotlib.pyplot  as plt
import numpy as np
import os
import nrrd
from utils import dice,sen,spe,pre

path =   '/media/s1/hxz/ICH/ICHTest/' 
pathdir = os.listdir(path)
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']='3'
model=torch.load('/media/s1/abnerhe/pytorch/ICHSeg/1A4/4A5/hdf5/Tol.pkl')
torch.cuda.empty_cache()
model.eval()

def normalize(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     x = (x-mean)/std
#     x = np.max(x)
#     x = np.min(x)
    M = np.max(x)
    N = np.min(x)
    X = (x-N)/(M-N)
    return x
x=np.zeros((1,512,512,1))
y=np.zeros((1,512,512,1))
Count=0
TolD=0
TolS=0
TolP=0
TolR=0
data=np.zeros((1,1,256,256))
label=np.zeros((1,256,256))
T=[]
sor=np.zeros((512,512))
with torch.no_grad():
    for t in range(97):
        if(pathdir[t]=='.ipynb_checkpoints'):
            t=t+1
        y1,i=nrrd.read(path+pathdir[t]+'/'+pathdir[t]+'seg.nrrd')
        print(pathdir[t])
        Count+=1
        x1,j=nrrd.read(path+pathdir[t]+'/'+pathdir[t]+'raw.nrrd')
        c=[np.nonzero(y1)]
        c=[[np.min(i,1),np.max(i,1)] for i in c]
        f=np.array([np.min([i[0] for i in c],0),np.max([i[1] for i in c],0)]).T
        x1[x1>100]=0
        x1[x1<0]=0
        x1=normalize(x1)
        c=x1.shape[2]
        P=np.zeros((c,512,512))
        for i in range(x1.shape[2]):
                data=np.zeros((1,1,512,512))
                data[0,0,:,:]=x1[:,:,i]
                data1=torch.Tensor(data)
                data1=data1.cuda()
                p = model(data1)
                p=p.cpu()
                s=p.data.numpy()
                P[i]=s
        print('P:'+str(Count))
        print(np.max(P))
        print(np.min(P))
        P[P>0.5]=1
        P[P<=0.5]=0
        T.append(P)
        pp=P.transpose(1,2,0)
        dice1=dice(pp,y1)
        sen1=sen(pp,y1)
        spe1=spe(pp,y1)
        pre1=pre(pp,y1)
        TolD=TolD+dice1
        TolS=TolS+sen1
        TolP=TolP+spe1
        TolR=TolR+pre1
        print(dice1)
        print(sen1)
        print(spe1)
        print(pre1)
        print(TolD/Count)
        print(TolS/Count)
        print(TolP/Count)
        print(TolR/Count)
        print(s.shape)
        plt.subplot(231)
        plt.imshow(pp[:,:,f[2,1]-2],cmap = plt.get_cmap('gray'))
        plt.subplot(232)
        plt.imshow(y1[:,:,f[2,1]-2],cmap = plt.get_cmap('gray'))
        plt.subplot(233)
        plt.imshow(x1[:,:,f[2,1]-2],cmap = plt.get_cmap('gray'))
        plt.subplot(234)
        plt.imshow(pp[:,:,0],cmap = plt.get_cmap('gray'))
        plt.subplot(235)
        plt.imshow(x1[:,:,0],cmap = plt.get_cmap('gray'))
        plt.show()
# t=s.argmax(axis=1)
print(TolD/Count)
print(TolS/Count)
print(TolP/Count)
print(TolR/Count)
# np.save('/media/s1/hxz/picture/seg92.npy',pp)
# np.save('/media/s1/hxz/picture/truth92.npy',y1)
# np.save('/media/s1/hxz//picture/raw92.npy',x1)
# plt.subplot(234)
# plt.imshow(data[0,2,:,:],cmap = plt.get_cmap('gray'))
# plt.subplot(235)
# plt.imshow(data[0,3,:,:],cmap = plt.get_cmap('gray'))

# from PIL import Image
# I=Image.fromarray(s[0,0,:,:])

# I.show()
