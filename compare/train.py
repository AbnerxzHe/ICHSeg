import sys
from optparse import OptionParser
import torch.backends.cudnn as cudnn
import torch.nn as nn
from keras.utils import to_categorical
from torch.utils.data import Dataset,DataLoader
import h5py
import  os
import random
from torch.autograd import Variable
import torch
import SimpleITK as sitk
import nrrd
import numpy as np
from model import UNet
from  metrics  import dice_score,pixelwise_acc,iou,sen_score,DiceLoss
from loss import CB_loss

torch.manual_seed(10)  

# os.environ['OMP_NUM_THREADS']='1'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
unet = UNet(n_channels=1, n_classes=1)
unet=unet.cuda()

print(unet)


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
path =  '/media/s1/hxz/ICH/ICHTrain/' 
x2=np.zeros((1,352,352))
y2=np.zeros((1,352,352))
bathsize = 24
class getDataset(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """
    #root = '/media/s1/hxz/DATA/19/'
    def __init__(self, root, augment=None):
        # 这个list存放所有图像的地址
        self.path = root  
        self.pathdir = os.listdir(self.path)
        self.Datapathdir=self.pathdir
        self.Data=len(self.pathdir)

    def __getitem__(self, index):
        tr=random.randint(0,20)
        if(self.Datapathdir[index]=='.ipynb_checkpoints'):
            index=index+1
        if(tr%2==0):
            x1,i=nrrd.read(self.path+self.Datapathdir[index]+'/'+self.Datapathdir[index]+'raw.nrrd')
            y1,i=nrrd.read(self.path+self.Datapathdir[index]+'/'+self.Datapathdir[index]+'seg.nrrd')
        if(tr%2==1):
            x1,i=nrrd.read(self.path+self.Datapathdir[index]+'/'+self.Datapathdir[index]+'Rraw.nrrd')
            y1,i=nrrd.read(self.path+self.Datapathdir[index]+'/'+self.Datapathdir[index]+'Rseg.nrrd')
        c=[np.nonzero(y1)]
        c=[[np.min(i,1),np.max(i,1)] for i in c]
        f=np.array([np.min([i[0] for i in c],0),np.max([i[1] for i in c],0)]).T
        if(x1.shape[2]-5<f[2,1]):
            f[2,1]=x1.shape[2]-5
        x1[x1>=100]=0
        x1[x1<=0]=0
        x1=normalize(x1)
        a = random.randint(f[0,0]-5,f[0,1]-5)
        if(a>160):
            a = random.randint(100,160)
        a1=a+352
        b = random.randint(f[1,0]-5,f[1,1]-5)
        if(b>160):
            b = random.randint(100,160)
        b1=b+352
        for i in range(bathsize):
            e1=random.randint(f[2,0],f[2,1])
            x2[0,:,:]= x1[a:a1,b:b1,e1]
            y2[0,:,:]=y1[a:a1,b:b1,e1]
            x = torch.Tensor(x2) 
            y = torch.Tensor(y2)
            return x,y
                    

    def __len__(self):
        # 返回图像的数量
        return self.Data

optimizer = torch.optim.SGD(unet.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.BCELoss()      # 预测值和真实值的误差计算公式 (均方差)
loss_func2 = DiceLoss() 

                                         
ship_train_dataset = getDataset(path)
# 利用dataloader读取我们的数据对象，并设定batch-size和工作现场
# batch_size=2

ship_train_loader = DataLoader(ship_train_dataset, batch_size=4, num_workers=32, shuffle=False)  
for epoch in range(300):
    epoch_loss = 0
    Diceloss = 0
    Sen = 0
    print('Starting epoch {}/{}.'.format(epoch + 1, 300))
    for i, item in enumerate(ship_train_loader):
        data, label = item
#         print(data.shape)
        data=data.cuda()
        label=label.cuda()
        prediction = unet(data)
        dice1 = dice_score(prediction, label)
        sen1 = sen_score(prediction, label)
        loss = loss_func(prediction, label)+loss_func2(prediction, label)     # 计算两者的误差

#         loss = CB_loss(label,prediction, [1],1,"sigmoid", 0.9999, 2.0) 
        Diceloss = Diceloss+dice1
        Sen = Sen+sen1
        epoch_loss = epoch_loss +loss
       # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size , loss))
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
       

    print('Epoch finished ! Loss: {}'.format(epoch_loss / i)+'   Dice_Loss: {}'.format(Diceloss / i)+'    Sen: {}'.format(Sen / i))
    torch.save(unet,'/media/s1/abnerhe/pytorch/ICHSeg/Batchsize/L/hdf5/BDR1.pkl')  
