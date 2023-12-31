import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import glob
import cv2

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim


from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset


from tool import draw_loss,get_model,get_train_data



def train(datas='sirst',save_name = 'hh2_sir',epoch_num = 500,batch_size_train = 3,save_frq = 1000,resume_name = None,resume_file = None):
    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

        return loss0, loss

    # ------- 2. set the directory of training dataset --------
    
    if(resume_name and resume_file):
        resume_dir = os.path.join(os.getcwd(), 'saved_models', resume_name, resume_file)
    else:
        resume_dir = None
        
    model_dir = os.path.join('saved_models', save_name + os.sep)
    print("resume dir:{}".format(resume_dir))
    print("save dir:{}".format(model_dir))
    if(not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    batch_size_val = 1
    train_num = 0
    val_num = 0
    
    iter_num = []
    iter_loss = []    
    
    tra_img_name_list, tra_lbl_name_list = get_train_data(datas)
    
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, drop_last=True) #shuffle=True 乱序

    # ------- 3. define model --------
    
    net = get_model(save_name)
    if (resume_dir):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(resume_dir), strict=True)
            net.cuda()
        else:
            net.load_state_dict(torch.load(resume_dir, map_location='cpu'), strict=True)
    else:
        if torch.cuda.is_available():
            net.cuda()
      

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            iter_num.append(ite_num)
            iter_loss.append(running_loss / ite_num4val)

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + save_name +"_bce_itr_%d.pth" % (ite_num))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
    
    draw_loss(iter_num,iter_loss,"loss_picture_{}_{}".format(save_name,datas),save_name,datas)

if __name__ == '__main__':
    # dataset:  IRSTD/NUDT-SIRST/sirst50
    train(datas="NUDT",save_name="MFSRNet_NUDT",epoch_num=500,save_frq=1000,batch_size_train = 3) #, resume_name = 'MFSRNet_NUDT',resume_file = "MFSRNet_NUDT_bce_itr_101000.pth"