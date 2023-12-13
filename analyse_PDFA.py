import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torch.optim as optim
from model.metrics import *
import numpy as np
from PIL import Image
import glob
import time
import cv2
from tqdm import tqdm
import torch.utils.data as Data
from utils.data import SirstDataset

#from thop import profile

from data_loader import oRescaleT
from data_loader import oToTensor
from data_loader import oToTensorLab
from data_loader import SalObjDataset

from tool import *

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')



def main(image_name_list,label_name_list,para_name,save_name,prediction_dir):

    # --------- 1. get image path and name ---------
    
    model_dir = os.path.join(os.getcwd(), 'saved_models', save_name , para_name)
    print("analysing:",model_dir)

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = image_name_list,
                                        lbl_name_list = label_name_list,
                                        transform = transforms.Compose([oRescaleT(320),
                                                                      oToTensorLab(flag=0)]))
                                        
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = get_model(save_name)
    

    if torch.cuda.is_available():
        
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        net = nn.DataParallel(net)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)
    
    PD_FA_metric = PD_FA_bins(bins=20)
    
    iou_metric.reset()
    nIoU_metric.reset()
    PD_FA_metric.reset()
    best_iou = 0
    best_nIoU = 0
    total_iou = 0
    total_niou = 0
    # t0 = 0.0
    seen = 0
    IoU = 0
    nIoU = 0
    #####################
    for i_test, data_test in enumerate(test_salobj_dataloader):
        seen += 1
        # print("inferencing:", image_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if prediction_dir is not None:
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(image_name_list[i_test],pred,prediction_dir)

        # iou/niou
        labels = data_test['label'].cpu()
        output = pred.unsqueeze(0).cpu()
        
        iou_metric.update(output, labels)
        nIoU_metric.update(output, labels)
        PD_FA_metric.update(output,labels)
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()
        PD, FA = PD_FA_metric.get()
        
        
        if IoU > best_iou:
            best_iou = IoU
        if nIoU > best_nIoU:
            best_nIoU = nIoU

        total_iou += IoU
        total_niou += nIoU

        del d1,d2,d3,d4,d5,d6,d7

    # IoU = total_iou / seen
    # nIoU = total_niou / seen
    print(IoU, nIoU) 
    print(best_iou, best_nIoU)
    print("PD_FA:",PD, FA)
    
    return IoU, nIoU, best_iou, best_nIoU,PD, FA


if __name__ == "__main__":
    best_iou = 0
    best_nioU = 0
    best_sum_iou = 0 
    best_1 = None
    best_2 = None
    best_3 = None
    
    best_pth = None
    
    # dataset:  /sirst/sirst_new/synthetic/IRSTD/NUDT-SIRST
    data_name = 'sirst50'
    image_name_list,label_name_list = get_test_data(data_name)
    print(f"---------------{data_name}----------------")
    # model:
    model_name = 'MFSRNet'
    save_name = model_name + '_' + 'nuaa'
    # dir:
    root_dir = os.listdir(os.path.join(os.getcwd(), 'saved_models', save_name))
    root_dir.sort()
    # prediction_dir = os.path.join(os.getcwd(), 'Predictions', data_name , model_name + os.sep)
    
    
    result = open(save_name +'sirst50.txt','a+')

    '''
    for pap in root_dir:
        x = pap.split('_')
        if(len(x) < 5):
            continue
        itern = x[5]
        print('---',itern,' iter')
        
        IoU, nIoU, _, _ = main(image_name_list,label_name_list,pap,save_name=save_name)
        
        if(IoU > best_iou):
            best_iou = IoU
            best_1 = itern
        if(nIoU > best_nioU):
            best_nioU = nIoU
            best_2 = itern
        if(nIoU + IoU > best_sum_iou):
            best_sum_iou = nIoU + IoU
            best_3 = itern
        
        print("best IoU = {}  ,{}iter".format(best_iou,best_1))
        print("best nIoU = {}  ,{}iter".format(best_nioU,best_2))
        print("best IoU+nIoU = {}  ,{}iter".format(best_sum_iou,best_3))

        result.write("{}iter: IoU={} nIoU={}\n".format(itern,IoU,nIoU))
    result.write("best IoU = {}  ,{}iter\n".format(best_iou,best_1))
    result.write("best nIoU = {}  ,{}iter\n".format(best_nioU,best_2))
    result.write("best IoU+nIoU = {}  ,{}iter\n".format(best_sum_iou,best_3))
    result.close()
    '''
    
    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,"uiunet_sir_sirst_bce_itr_53000_train_0.004350_tar_0.000161.pth",save_name=save_name,prediction_dir=prediction_dir)
    
    for pth_id,pap in enumerate(root_dir):
        x = pap.split('_')
        if(len(x) < 5):
            continue
        itern = x[5]
        print('---',itern,' iter')
        IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,pap,save_name=save_name,prediction_dir=None)
        if(IoU > best_iou):
              best_iou = IoU
              best_1 = itern
        if(nIoU > best_nioU):
            best_nioU = nIoU
            best_2 = itern
        if(nIoU + IoU > best_sum_iou):
            best_sum_iou = nIoU + IoU
            best_3 = itern
            best_pth = pth_id
        
        print("best IoU = {}  ,{}iter".format(best_iou,best_1))
        print("best nIoU = {}  ,{}iter".format(best_nioU,best_2))
        print("best IoU+nIoU = {}  ,{}iter".format(best_sum_iou,best_3))

        result.write("{}iter: IoU={} nIoU={}\n".format(itern,IoU,nIoU))
        result.write("{}iter: PD={} FA={}\n".format(itern,PD,FA))
        
    result.write("best nIoU = {}  ,{}iter\n".format(best_nioU,best_2))
    result.write("best IoU+nIoU = {}  ,{}iter\n".format(best_sum_iou,best_3))
    result.close()
    
    
    prediction_dir = os.path.join(os.getcwd(), 'Predictions', data_name , model_name+"_newresults" + os.sep)
    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,'hhl6con_IRSTD_IRSTD_bce_itr_29000_train_0.003714_tar_0.000249.pth',save_name=save_name,prediction_dir=prediction_dir)
    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,'hhl61_IRSTD2_IRSTD_bce_itr_116000_train_0.002761_tar_0.000025.pth',save_name=save_name,prediction_dir=prediction_dir)
    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,'RH652_sirst50_sirst50_bce_itr_38000_train_0.004062_tar_0.000072.pth',save_name=save_name,prediction_dir=prediction_dir)
    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,'u2conv_sirst50_sirst50_bce_itr_38000_train_0.004020_tar_0.000079.pth',save_name=save_name,prediction_dir=prediction_dir)
    IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,root_dir[best_pth],save_name=save_name,prediction_dir=prediction_dir)

    #IoU, nIoU, _, _,PD, FA = main(image_name_list,label_name_list,"hhl61_sirst50_sirst50_bce_itr_40000_train_0.004160_tar_0.000079.pth",save_name=save_name,prediction_dir=prediction_dir)
    
        
    
