import torch
import torch.nn as nn

def Conv1x1(in_ch,out_ch,padding=0):
    conv = nn.Sequential(
        nn.Conv2d(in_ch,out_ch,1,padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return conv

def Conv3x3(in_ch,out_ch,padding=1):
    conv = nn.Sequential(
        nn.Conv2d(in_ch,out_ch,3,padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return conv



class AdaptiveFuse(nn.Module):
    def __init__(self,fnum, in_ch:list, mid_ch:list, out_ch):
        super(AdaptiveFuse,self).__init__()
        self.fnum = fnum
        if(len(in_ch)!=self.fnum):
            raise KeyError('The parameter \'in_ch\' is wrong')
        if(len(mid_ch)!=self.fnum):
            raise KeyError('The parameter \'mid_ch\' is wrong')
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.reduct = nn.ModuleList([Conv1x1(in_ch[i],mid_ch[i])for i in range(self.fnum)])
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveMaxPool2d(1)
        self.mid_sum = sum(mid_ch)
        # self.conv1 = Conv1x1(self.mid_sum*2,self.mid_sum)
        self.conv1 = nn.Conv2d(self.mid_sum*2,self.mid_sum,1,padding=0)
        self.out_neck = Conv1x1(self.mid_sum,out_ch)



    def forward(self, features):
        if(len(features)!=self.fnum):
            raise KeyError('Input features dim is wrong')
        hx = torch.cat([self.reduct[i](features[i]) for i in range(self.fnum)],dim = 1)
        x_avg = self.pool_1(hx)
        x_max = self.pool_2(hx)
        xw = self.conv1(torch.cat([x_avg,x_max],dim = 1))
        hx = hx * xw
        x_out = self.out_neck(hx)
        return x_out

