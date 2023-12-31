# coding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import cv2
import numpy as np
import glob

from model import *


def get_test_data(data_name):
    if (data_name == "sirst50"):
        test_txt = "sirst/list/test_NUAA-SIRST.txt"
        img_dir = "sirst/images/"
        lbl_dir = "sirst/masks/"
        with open(test_txt) as f:
            lines = f.readlines()
            need = [i.strip().split('_')[1] for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('_')[1].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('_')[1]] = i

        image_name_list = []
        label_name_list = []
        for i in need:
            image_name_list.append(img_dict[i])
            label_name_list.append(lbl_dict[i])

    elif (data_name == 'IRSTD'):
        train_txt = "IRSTD-1k/test.txt"
        img_dir = "IRSTD-1k/IRSTD1k_Img/"
        lbl_dir = "IRSTD-1k/IRSTD1k_Label/"
        with open(train_txt) as f:
            lines = f.readlines()
            need = [i.strip() for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        print()
        for i in img_name_list:
            img_dict[i.split('//')[1].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('//')[1].split('.')[0]] = i

        image_name_list = []
        label_name_list = []
        for i in need:
            image_name_list.append(img_dict[i])
            label_name_list.append(lbl_dict[i])


    elif (data_name == 'NUDT-SIRST'):
        train_txt = "NUDT-SIRST/test_NUDT-SIRST.txt"
        img_dir = "NUDT-SIRST/images/"
        lbl_dir = "NUDT-SIRST/masks/"
        with open(train_txt) as f:
            lines = f.readlines()
            need = [i.strip() for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i[-10:].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i[-10:].split('.')[0]] = i

        image_name_list = []
        label_name_list = []
        for i in need:
            image_name_list.append(img_dict[i])
            label_name_list.append(lbl_dict[i])
    else:
        raise KeyError("dataset name is wrong")

    return image_name_list, label_name_list


def get_train_data(data_name):
    if (data_name == 'sirst50'):
        test_txt = "sirst/list/train_NUAA-SIRST.txt"
        img_dir = "sirst/images/"
        lbl_dir = "sirst/masks/"
        with open(test_txt) as f:
            lines = f.readlines()
            need = [i.strip().split('_')[1] for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('_')[1].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('_')[1]] = i

        tra_img_name_list = []
        tra_lbl_name_list = []
        for i in need:
            tra_img_name_list.append(img_dict[i])
            tra_lbl_name_list.append(lbl_dict[i])
            
    elif (data_name == 'IRSTD'):
        train_txt = "IRSTD-1k/trainval.txt"
        img_dir = "IRSTD-1k/IRSTD1k_Img/"
        lbl_dir = "IRSTD-1k/IRSTD1k_Label/"
        with open(train_txt) as f:
            lines = f.readlines()
            need = [i.strip() for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('/')[2].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('/')[2].split('.')[0]] = i

        tra_img_name_list = []
        tra_lbl_name_list = []
        for i in need:
            tra_img_name_list.append(img_dict[i])
            tra_lbl_name_list.append(lbl_dict[i])
    elif (data_name == 'NUDT-SIRST'):
        train_txt = "NUDT-SIRST/train_NUDT-SIRST.txt"
        img_dir = "NUDT-SIRST/images/"
        lbl_dir = "NUDT-SIRST/masks/"
        with open(train_txt) as f:
            lines = f.readlines()
            need = [i.strip() for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('/')[2].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('/')[2].split('.')[0]] = i

        tra_img_name_list = []
        tra_lbl_name_list = []
        for i in need:
            tra_img_name_list.append(img_dict[i])
            tra_lbl_name_list.append(lbl_dict[i])
    else:
        raise KeyError("dataset name is wrong")

    return tra_img_name_list, tra_lbl_name_list


def get_model(save_name):
    if (save_name[:5] == 'u2net'):
        print("net : U2NET")
        net = U2NET(3, 1)
    elif(save_name[:4]=="MFSR"):
        print("net : MFSRNET")
        net = MFSRNet(3, 1)
    elif(save_name[:6] == 'u2conv'):
        print("net : U2NET_conv")
        net = U2NET_conv(3, 1)
    elif(save_name[:5] == 'halfb'):
        print("net : halfbranch")
        net = halfbranch(3, 1)
    elif(save_name[:5] == 'woAFF'):
        print("net : woAFF")
        net = woAFF(3, 1)
    elif(save_name[:4] == 'woDB'):
        print("net : woDB")
        net = woDB(3, 1)
    elif(save_name[:4] == 'woSR'):
        print("net : woSR")
        net = woSR(3, 1)
    elif (save_name[:6] == 'uiunet'):
        print("net : UIUNET")
        net = UIUNET(3, 1)
    
    else:
        raise KeyError("net not exist")

    return net


def draw_loss(x, y, save_name, model_name, data_name):
    fig = plt.figure(figsize=(20, 20))  # figsize是图片的大小
    plt.plot(x, y, 'g-', label=u'U2net(SIRST)')
    plt.legend(prop={"size": 20})
    plt.xlabel(u'iters', fontdict={"size": 25})
    plt.ylabel(u'loss', fontdict={"size": 25})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('loss for {} on {} in training'.format(model_name, data_name), fontsize=30)
    fig.savefig('./saved_models/training/{}.png'.format(save_name))


def look_up(fn):
    model = torch.load(fn, map_location='cpu')

    for i in model:
        print(i, model[i].shape)


def split_map(image_list, savepath):
    for img_dir in image_list:
        print(img_dir)
        mask = cv2.imread(img_dir, 0)
        body = cv2.blur(mask, ksize=(5, 5))  # 先去除噪点
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        # body = body**0.5

        tmp = body[np.where(body > 0)]
        if len(tmp) != 0:
            body[np.where(body > 0)] = np.floor(tmp / np.max(tmp) * 255)  # 归一化

        '''if not os.path.exists(datapath+'/body-origin/'):
            os.makedirs(datapath+'/body-origin/')
        cv2.imwrite(datapath+'/body-origin/'+name, body)'''

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        imagex = img_dir.split('_')
        name = imagex[0][11:] + '_' + imagex[1] + '.png'
        print(f"generating the detail for {name}")
        cv2.imwrite(savepath + name, mask - body)


def get_imagelist(datas):
    if (datas == 'synthetic'):
        data_dir = os.path.join(os.getcwd(), 'data', 'training')
        data_dir += '/'
        image_ext = '1.png'
        label_ext = '2.png'
        tra_img_name_list = glob.glob(data_dir + '*' + image_ext)
        tra_lbl_name_list = glob.glob(data_dir + '*' + label_ext)
        tra_img_name_list.sort()
        tra_lbl_name_list.sort()
        return tra_img_name_list, tra_lbl_name_list

    elif (datas == 'sirst'):
        test_txt = os.path.join("sirst", "idx_427", "trainval.txt")
        img_dir = "sirst/images/"
        lbl_dir = "sirst/masks/"
        with open(test_txt) as f:
            lines = f.readlines()
            need = [i.strip().split('_')[1] for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('_')[1].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('_')[1]] = i

        tra_img_name_list = []
        tra_lbl_name_list = []
        for i in need:
            tra_img_name_list.append(img_dict[i])
            tra_lbl_name_list.append(lbl_dict[i])
        return tra_img_name_list, tra_lbl_name_list

    elif (datas == 'sirst_all'):
        img_dir = "sirst/images"
        lbl_dir = "sirst/masks"
        imgs = os.listdir(img_dir)
        lbls = os.listdir(lbl_dir)
        tra_img_name_list = []
        tra_lbl_name_list = []
        for img_name in imgs:
            lbl_name = img_name.split('.')[0] + '_pixels0.png'
            if (lbl_name in lbls):
                tra_img_name_list.append(os.path.join(img_dir, img_name))
                tra_lbl_name_list.append(os.path.join(lbl_dir, lbl_name))
        return tra_img_name_list, tra_lbl_name_list
        
    elif (data_name == 'NUDT-SIRST'):
        train_txt = "NUDT-SIRST/train_NUDT-SIRST.txt"
        img_dir = "NUDT-SIRST/images/"
        lbl_dir = "NUDT-SIRST/masks/"
        with open(train_txt) as f:
            lines = f.readlines()
            need = [i.strip() for i in lines]

        img_name_list = glob.glob(img_dir + '*.png')
        lbl_name_list = glob.glob(lbl_dir + '*.png')
        img_dict = dict()
        lbl_dict = dict()
        for i in img_name_list:
            img_dict[i.split('/')[2].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('/')[2].split('.')[0]] = i

        tra_img_name_list = []
        tra_lbl_name_list = []
        for i in need:
            tra_img_name_list.append(img_dict[i])
            tra_lbl_name_list.append(lbl_dict[i])
        return tra_img_name_list, tra_lbl_name_list

    else:
        print("wrong dataset")
        return

