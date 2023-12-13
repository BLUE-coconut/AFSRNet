# coding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import cv2
import numpy as np
import glob

from model import *
from BasicIRSTD.model import *


def get_test_data(data_name):
    if (data_name == "sirst"):
        test_txt = "/home/hehaolan/Multilevel_U2net_ISTD/BasicIRSTD/datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST_20.txt"
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
    elif (data_name == "sirst50"):
        test_txt = "/home/hehaolan/Multilevel_U2net_ISTD/BasicIRSTD/datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST_50.txt"
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
    elif (data_name == "sirst_new"):
        data_dir = "test_data/Quantification_Results/test_images"
        dirs = os.listdir(data_dir)
        img_dir = "test_data/Quantification_Results/test_images"
        lbl_dir = "test_data/Quantification_Results/test_labels"
        image_name_list = []
        label_name_list = []
        for img_name in dirs:
            image_name_list.append(os.path.join(img_dir, img_name))
            label_name_list.append(os.path.join(lbl_dir, img_name.split('.')[0] + '_pixels0.png'))

    elif (data_name == "synthetic"):
        data_dir = "data/test_gt"
        dirs = os.listdir(data_dir)
        img_dir = "data/test_org/"
        lbl_dir = "data/test_gt/"
        image_name_list = []
        label_name_list = []
        for img_name in dirs:
            image_name_list.append(os.path.join(img_dir, img_name))
            label_name_list.append(os.path.join(lbl_dir, img_name))

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
        for i in img_name_list:
            img_dict[i.split('/')[2].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('/')[2].split('.')[0]] = i

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
            img_dict[i.split('/')[2].split('.')[0]] = i
        for i in lbl_name_list:
            lbl_dict[i.split('/')[2].split('.')[0]] = i

        image_name_list = []
        label_name_list = []
        for i in need:
            image_name_list.append(img_dict[i])
            label_name_list.append(lbl_dict[i])
    else:
        raise KeyError("dataset name is wrong")

    return image_name_list, label_name_list


def get_train_data(data_name):
    if (data_name == 'synthetic'):
        data_dir = os.path.join(os.getcwd(), 'data', 'training')
        data_dir += '/'
        image_ext = '1.png'
        label_ext = '2.png'
        tra_img_name_list = glob.glob(data_dir + '*' + image_ext)
        tra_lbl_name_list = glob.glob(data_dir + '*' + label_ext)
        tra_img_name_list.sort()
        tra_lbl_name_list.sort()
    elif (data_name == 'sirst'):
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
    elif (data_name == 'sirst50'):
        test_txt = "/home/hehaolan/Multilevel_U2net_ISTD/BasicIRSTD/datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST_50.txt"
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
    elif (data_name == 'sirst_all'):
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
    elif(save_name[:5] == 'woadf'):
        print("net : woADF")
        net = woADF(3, 1)
    elif(save_name[:4] == 'woDB'):
        print("net : woDB")
        net = woDB(3, 1)
    
    elif (save_name[:6] == 'uiunet'):
        print("net : UIUNET")
        net = UIUNET(3, 1)
    elif(save_name[:5] == 'ISNet'):
        print("net : ISNet")
        net = ISNet(mode=mode)
    elif(save_name[:5] == 'ISTDU'):
        print("net : ISTDU-Net")
        net = ISTDU_Net(3)
    elif(save_name[:5] == 'RDIAN'):
        print("net : ISNet")
        net = RDIAN()
    elif(save_name[:3] == 'ACM'):
        print("net : ACMNet")
        net = ACM()
    elif(save_name[:3] == 'ALC'):
        print("net : ALCNet")
        net = ALCNet()
    elif(save_name[:3] == 'DNA'):
        print("net : DNANet")
        net = DNANet(mode=mode)


    elif (save_name[:5] == 'hhl30'):
        print("net : hhl30")
        net = UIUNET_New(3, 1)
    elif (save_name[:5] == 'hhl31'):
        print("net : hhl31")
        net = HHL3_1(3, 1)
    elif (save_name[:5] == 'hhl32'):
        print("net : hhl32")
        net = HHL3_2(3, 1)
    elif (save_name[:4] == 'hhl4'):
        print("net : hhl4")
        net = UIUNET_hhl4(3, 1)



    elif (save_name[:5] == 'hhl54'):
        print("net : hhl54")
        net = HHL5_RH65_ADF_4(3, 1)
    elif (save_name[:5] == 'hhl55'):
        print("net : hhl55")
        net = HHL5_RH65_ADF_5(3, 1)
    elif (save_name[:5] == 'hhl56'):
        print("net : hhl56")
        net = HHL5_RH65_ADF_6(3, 1)
    elif (save_name[:5] == 'hhl51'):
        print("net : hhl51")
        net = HHL5_RH65_ADF_2(3, 1)
    elif (save_name[:5] == 'hhl52'):
        print("net : hhl52")
        net = HHL5_RH65_ADF_3(3, 1)
    elif (save_name[:5] == 'hhl50'):
        print("net : hhl50")
        net = HHL5_RH65_ADF_0(3, 1)
    elif (save_name[:4] == 'hhl5'):
        print("net : hhl5")
        net = HHL5_RH65_ADF_(3, 1)

    elif(save_name[:5] == 'hhl61'):
        print("net : hhl61")
        net = HHL6_RH65_ADF_1(3, 1)

    elif(save_name[:7] == 'hhl6con'):
        print("net : hhl6con")
        net = HHL6_RH65_ADF_con6(3, 1)
    elif(save_name[:7] == 'hhl6nor'):
        print("net : hhl6nor")
        net = HHL6_RH65_ADF_1_norm(3, 1)

    elif (save_name[:5] == 'hhl62'):
        print("net : hhl62")
        net = HHL6_RH65_ADF_2(3, 1)
    elif (save_name[:5] == 'hhl63'):
        print("net : hhl63")
        net = HHL6_RH65_ADF_3(3, 1)
    elif (save_name[:5] == 'hhl65'):
        print("net : hhl65")
        net = HHL6_RH65_ADF_5(3, 1)
    elif (save_name[:5] == 'hhl66'):
        print("net : hhl66")
        net = HHL6_RH65_ADF_6(3, 1)

    elif (save_name[:4] == 'UCF0'):
        print("net : UCF_NET0")
        net = UCF_NET(3, 1)
    elif (save_name[:7] == 'UCF1nor'):
        print("net : UCFADF_norm_NET")
        net = UCFADF_norm_NET(3, 1)
    elif (save_name[:4] == 'UCF1'):
        print("net : UCFADF_NET")
        net = UCFADF_NET(3, 1)
    elif (save_name[:5] == 'RH650'):
        print("net :RH_650")
        net = RH_65_0(3, 1)
    elif (save_name[:5] == 'RH651'):
        print("net : RH_65_1_cor")
        net = RH_65_1(3, 1)
    elif (save_name[:5] == 'RH652'):
        print("net : RH_652")
        net = RH_65_2(3, 1)
    elif (save_name[:4] == 'RH65'):
        print("net : RH_65")
        net = UIUNET_New_RH_65Image(3, 1)
    elif (save_name[:5] == 'UCFRH'):
        print("net : UCF_RH65_cor")
        net = UCF_RH65_cor(3, 1)

    elif (save_name[:4] == 'RH35'):
        print("net : UIUNET_New_RH_35")
        net = UIUNET_New_RH_35Image(3, 1)
    elif (save_name[:4] == 'RH45'):
        print("net : UIUNET_New_RH_45")
        net = UIUNET_New_RH_45Image(3, 1)

    elif (save_name[:4] == 'RQ44'):
        print("net : UIUNET_New_RQ_44")
        net = UIUNET_New_RQ_44Image(3, 1)
    elif (save_name[:4] == 'DR66'):
        print("net : UIUNET_New_DR_66")
        net = UIUNET_New_DR_66Image(3, 1)
    elif (save_name[:6] == 'DRH665'):
        print("net : UIUNET_New_DRH_665")
        net = UIUNET_New_DRH_665Image(3, 1)

    elif (save_name[:6] == 'RHmt20'):
        print("net : UIUNET_Mutual_RH_65Image2_0")
        net = UIUNET_Mutual_RH_65Image2_0(3, 1)
    elif (save_name[:6] == 'RHmt21'):
        print("net : UIUNET_Mutual_RH_65Image2_1")
        net = UIUNET_Mutual_RH_65Image2_1(3, 1)
    elif (save_name[:4] == 'RHmt'):
        print("net : UIUNET_Mutual_RH_65Image")
        net = UIUNET_Mutual_RH_65Image(3, 1)

    elif (save_name[:4] == 'Udet'):
        print("net : UIUNET_Udetail")
        net = UIUNET_Udetail(3, 1)

    elif(save_name[:4]=='3SFR'):
        print("net : Dual_ADF_3SFR_1")
        net = Dual_ADF_3SFR_1(3,1)
    elif(save_name[:4]=='UCF3'):
        print("net : UCF_3SFR_NET")
        net = UCF_3SFR_NET(3,1)

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


'''
if __name__=='__main__':
    _ , lbl_list = get_imagelist('sirst_all')

    split_map(image_list=lbl_list,savepath="sirst/details/")
#look_up("./saved_models/hhl3_sir/hhl3_sir_sirst_bce_itr_52000_train_0.004080_tar_0.000106.pth")'''