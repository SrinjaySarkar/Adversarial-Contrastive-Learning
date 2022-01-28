import torch
import os
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils 
from torchvision import transforms

from models import resnet,basic_block

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_load_path="/vinai/sskar/CSS47_robust_representation/contrast/checkpoint/ckpt.t7sample_42"
"""In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files."""

root_path="/vinai/sskar/CIFAR-10-C"
c_type="contrast"
load_path=os.path.join(root_path,c_type) + ".npy"
x=np.load(load_path).astype(np.float32)
print(x.shape)

class c_cifar(utils.data.Dataset):
    def __init__(self,root_path,c_type,test=True):
        super(c_cifar,self).__init__()
        self.root_path=root_path
        self.c_type=c_type
        self.test=test
        self.transform=transforms.ToTensor()
        load_path=os.path.join(self.root_path,c_type) + ".npy"
        label_load_path=os.path.join(self.root_path,"labels.npy")
        assert (os.path.exists(load_path)) , "corruption type does not exist"
    
    def __getitem__(self,idx):
        data=np.load(os.path.join(self.root_path,self.c_type) + ".npy").astype(np.float32)
        # print(data)
        # data=data.reshape(50000,32,32,3)
        labels=np.load(os.path.join(self.root_path,"labels.npy"))
        # labels=labels.reshape(50000,)
        if self.test:
            data=data[:10000,:,:,:]
            labels=labels[:10000]
        else:
            data=data
            labels=labels
        data_sample=data[idx]
        label_sample=labels[idx]
        return (data_sample,label_sample)
    
    def __len__(self):
        if self.test:
            return (10000)
        else:
            return (50000)       

#dataloader
sample_path="/vinai/sskar/CIFAR-10-C"
corrupt_dataset=c_cifar(root_path=sample_path,c_type="gaussian_noise",test=True)
bs=32
n_classes=10
corrupt_dataloader=utils.data.DataLoader(corrupt_dataset,batch_size=bs,shuffle=True)

#model
model=resnet(basic_block,[2,2,2,2],n_classes=n_classes,contrast=True).to(device)
expansion=1
linear=torch.nn.Sequential(torch.nn.Linear(512*expansion,10)).to(device)

checkpoint_=torch.load(model_load_path)
new_state_dict=OrderedDict()
for k,v in checkpoint_['model'].items():
    name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

linearcheckpoint_ = torch.load(model_load_path+'_linear')
new_state_dict = OrderedDict()
for k, v in linearcheckpoint_['model'].items():
    name = k
    new_state_dict[name] = v
linear.load_state_dict(new_state_dict)

criterion=torch.nn.CrossEntropyLoss()
model.eval()
linear.eval()
test_clean_loss=0
test_adv_loss=0
clean_correct=0
adv_correct=0
clean_acc=0
total=0


#testing
for batch,(image,label) in enumerate(corrupt_dataloader):
    label=label.type(torch.LongTensor)
    image=image.permute(0,3,1,2)
    img,y=image.to(device),label.to(device)
    total+=y.size(0)
    # np.save()
    # print(img.dtype)
    # print(y.dtype)
    out=linear(model(img))
    _,predx=torch.max(out.data,1)
    clean_loss=criterion(out,y)
    clean_correct+=predx.eq(y.data).cpu().sum().item()
    clean_acc=100.*clean_correct/total
    print(clean_acc)
print("done")
clean_acc=100.*clean_correct/total
print(clean_acc)