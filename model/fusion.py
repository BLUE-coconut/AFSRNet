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



class Gate(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Gate,self).__init__()
        # relu+bn
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_in = nn.Conv2d(in_ch,in_ch,3,padding=1)
        self.conv_weight = nn.Conv2d(2,in_ch,1,padding=0)
        self.conv_out = Conv1x1(in_ch = in_ch,out_ch = out_ch)

    def forward(self,x):
        mx = self.conv_in(x)
        max_result,_=torch.max(mx,dim=1,keepdim=True)
        avg_result=torch.mean(mx,dim=1,keepdim=True)
        x_ma = torch.cat([max_result,avg_result],dim=1)
        #x_out = self.conv_out(x_ma)
        x_out = self.conv_out((self.conv_weight(x_ma))*x)

        return x_out

class Gate2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Gate2,self).__init__()
        # sigmoid
        self.in_ch = in_ch
        self.conv_in = nn.Conv2d(in_ch,in_ch,3,padding=1)
        self.conv_weight = nn.Conv2d(2,in_ch,1,padding=0)
        self.sig = nn.Sigmoid()

        self.conv_out = nn.Conv2d(in_ch,out_ch,1,padding=0)
        self.bn_out = nn.BatchNorm2d(out_ch)

    def forward(self,x):
        mx = self.conv_in(x)
        max_result,_=torch.max(mx,dim=1,keepdim=True)
        avg_result=torch.mean(mx,dim=1,keepdim=True)
        x_ma = torch.cat([max_result,avg_result],dim=1)

        weight = self.conv_weight(x_ma)
        x_weighted = self.sig(weight)*x
        x_out = self.bn_out(self.conv_out(x_weighted))

        return x_out

class Gate3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Gate3,self).__init__()
        # sigmoid; without GAvg&GMax
        self.in_ch = in_ch
        self.conv_in = nn.Conv2d(in_ch,in_ch,3,padding=1)
        self.conv_weight = nn.Conv2d(in_ch,in_ch,1,padding=0)
        self.sig = nn.Sigmoid()

        self.conv_out = Conv1x1(in_ch,out_ch)

    def forward(self,x):
        mx = self.conv_in(x)
        #max_result,_=torch.max(mx,dim=1,keepdim=True)
        #avg_result=torch.mean(mx,dim=1,keepdim=True)
        #x_ma = torch.cat([max_result,avg_result],dim=1)

        weight = self.conv_weight(mx)
        x_weighted = self.sig(weight)*x
        x_out = self.conv_out(x_weighted)

        return x_out
    
class CAM(nn.Module):
    def __init__(self,low_ch,high_ch,out_ch,r=2):
        super(CAM,self).__init__()
        # x_raw,x_down通道数一致
        self.high_ch = high_ch
        self.low_ch = low_ch
        self.out_ch = out_ch

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.weight1 = Conv1x1(self.high_ch,(self.high_ch+self.low_ch)//r)
        self.weight2 = Conv1x1((self.high_ch+self.low_ch)//r,self.low_ch)

        self.weight3 = Conv1x1(self.low_ch,(self.high_ch+self.low_ch)//r)
        self.weight4 = Conv1x1((self.high_ch+self.low_ch)//r,self.high_ch)

        #self.fuse = Conv1x1(self.high_ch+self.low_ch,(self.high_ch+self.low_ch)//r)
        self.fuse2 = Conv1x1((self.high_ch+self.low_ch)//r*2+(self.high_ch+self.low_ch),self.out_ch)
        #self.fuse2 = Conv1x1((self.high_ch+self.low_ch)//r*3,self.out_ch)


    def forward(self,x_low,x_high):
        px1 = self.pool1(x_low)
        px2 = self.pool2(x_high)
        
        w1 = self.weight2(self.weight1(px2))
        w2 = self.weight4(self.weight3(px1))

        #x_fuse = self.fuse(torch.cat([x_low,x_high],dim=1))

        trix = self.fuse2(torch.cat([torch.cat([x_low,x_high],dim=1),w1*x_low,w2*x_high],dim=1))
        #trix = self.fuse2(torch.cat([x_fuse,w1*x_low,w2*x_high],dim=1))

        return trix
        

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
        self.conv1 = Conv1x1(self.mid_sum*2,self.mid_sum)
        # self.conv1 = nn.Conv2d(sum(mid_ch)*2,out_ch,1,padding=0)
        self.out_neck = nn.Conv2d(self.mid_sum,out_ch,1,padding=0)


    def forward(self, features):
        if(len(features)!=self.fnum):
            raise KeyError('Input features dim is wrong')
        hx = torch.cat([self.reduct[i](features[i]) for i in range(self.fnum)],dim = 1)
        x_max = self.pool_1(hx)
        x_avg = self.pool_2(hx)
        xw = self.conv1(torch.cat([x_max,x_avg],dim = 1))
        hx = hx * xw
        x_out = self.out_neck(hx)
        return x_out

class AdaptiveFuse2(nn.Module):
    def __init__(self,fnum, in_ch:list, mid_ch:list, out_ch):
        super(AdaptiveFuse2,self).__init__()
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
        x_max = self.pool_1(hx)
        x_avg = self.pool_2(hx)
        xw = self.conv1(torch.cat([x_max,x_avg],dim = 1))
        hx = hx * xw
        x_out = self.out_neck(hx)
        return x_out

class ADF(nn.Module):
    def __init__(self, in_ch:int, out_ch:int):
        super(ADF,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveMaxPool2d(1)
        self.conv1 = Conv1x1(self.in_ch*2,self.in_ch)
        # self.conv1 = nn.Conv2d(self.in_ch*2,self.in_ch,1,padding=0)
        self.out_neck = Conv1x1(in_ch,out_ch)


    def forward(self, x):
        x_max = self.pool_1(x)
        x_avg = self.pool_2(x)
        xw = self.conv1(torch.cat([x_max,x_avg],dim = 1))
        x_out = self.out_neck(x * xw)
        return x_out

class Mutual_AdaptiveFuse(nn.Module):
    def __init__(self,fnum:list, in_ch:list, mid_ch:list, out_ch:int,r=2):
        super(Mutual_AdaptiveFuse,self).__init__()
        self.fnum1 = fnum[0]
        self.fnum2 = fnum[1]
        if(self.fnum1!=self.fnum2+1):
            # raw image level should be 1 larger than 1/2 level
            # so that they can mustually interact with each other
            raise KeyError('The parameter \'fnum\' is wrong')
        if(len(in_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'in_ch\' is wrong')
        if(len(mid_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'mid_ch\' is wrong')
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch


        self.reduct = nn.ModuleList([Conv1x1(in_ch[i],mid_ch[i])for i in range(self.fnum1+self.fnum2)])
        self.Gate = nn.ModuleList([Gate(mid_ch[i+self.fnum1],mid_ch[i])for i in range(self.fnum2)])
        self.MidConv = nn.ModuleList([Conv3x3(mid_ch[i],mid_ch[i])for i in range(self.fnum2)])
        self.Cam = nn.ModuleList([CAM(mid_ch[i],mid_ch[i+1],mid_ch[i+1],r=r)for i in range(self.fnum1-1)])

        self.adf = ADF(sum(mid_ch[1:self.fnum1])+mid_ch[0],out_ch)

    def forward(self,features):
        if(len(features)!=self.fnum1+self.fnum2):
            raise KeyError('Input features dim is wrong')
        hx = [self.reduct[i](features[i]) for i in range(self.fnum1+self.fnum2)]
        hxfused = []
        hxfused.append(hx[0])
        for i in range(self.fnum2):
            Halfx = self.Gate[i](hx[i+self.fnum1])
            downx = hxfused[i]
            fusex = downx * self.MidConv[i](downx+Halfx)
            fsx = self.Cam[i](fusex,hx[i+1])
            hxfused.append(fsx)

        x_out = self.adf(torch.cat(hxfused,dim=1))

        return x_out

class Mutual_AdaptiveFuseRaw(nn.Module):
    def __init__(self,fnum:list, in_ch:list, mid_ch:list, out_ch:int,r=2):
        super(Mutual_AdaptiveFuseRaw,self).__init__()
        self.fnum = fnum
        if(len(in_ch)!=self.fnum):
            raise KeyError('The parameter \'in_ch\' is wrong')
        if(len(mid_ch)!=self.fnum):
            raise KeyError('The parameter \'mid_ch\' is wrong')
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch


        self.reduct = nn.ModuleList([Conv1x1(in_ch[i],mid_ch[i])for i in range(self.fnum)])
        self.Cam = nn.ModuleList([CAM(mid_ch[self.fnum-1-i],mid_ch[self.fnum-2-i],mid_ch[self.fnum-2-i],r=r)for i in range(self.fnum-1)])

        self.adf = ADF(sum(mid_ch),out_ch)

    def forward(self,features):
        if(len(features)!=self.fnum):
            raise KeyError('Input features dim is wrong')
        hx = [self.reduct[i](features[i]) for i in range(self.fnum)]
        hxfused = []
        hxfused.append(hx[self.fnum-1])

        for i in range(self.fnum-1):
            downx = hxfused[i]
            fsx = self.Cam[i](downx,hx[self.fnum-i-2])
            hxfused.append(fsx)

        x_out = self.adf(torch.cat(hxfused,dim=1))

        return x_out
    
class Mutual_AdaptiveFuseUp(nn.Module):
    def __init__(self,fnum:list, in_ch:list, mid_ch:list, out_ch:int,r=2,gate=0):
        super(Mutual_AdaptiveFuseUp,self).__init__()
        self.fnum1 = fnum[0]
        self.fnum2 = fnum[1]
        if(self.fnum1!=self.fnum2+1):
            # raw image level should be 1 larger than 1/2 level
            # so that they can mustually interact with each other
            raise KeyError('The parameter \'fnum\' is wrong')
        if(len(in_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'in_ch\' is wrong')
        if(len(mid_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'mid_ch\' is wrong')
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch


        self.reduct = nn.ModuleList([Conv1x1(in_ch[i],mid_ch[i])for i in range(self.fnum1+self.fnum2)])
        if(gate == 1):
            self.Gate = nn.ModuleList([Gate(mid_ch[self.fnum1+self.fnum2-1-i],mid_ch[self.fnum1-1-i])for i in range(self.fnum2)])
        elif(gate == 2):
            self.Gate = nn.ModuleList([Gate2(mid_ch[self.fnum1+self.fnum2-1-i],mid_ch[self.fnum1-1-i])for i in range(self.fnum2)])
        elif (gate == 3):
            self.Gate = nn.ModuleList([Gate3(mid_ch[self.fnum1 + self.fnum2 - 1 - i], mid_ch[self.fnum1 - 1 - i]) for i in range(self.fnum2)])
        elif (gate == 0):
            self.Gate = nn.ModuleList([Conv3x3(mid_ch[self.fnum1 + self.fnum2 - 1 - i], mid_ch[self.fnum1 - 1 - i]) for i in range(self.fnum2)])
        else:
            raise KeyError('parameter \'gate\' is wrong')
        self.MidConv = nn.ModuleList([Conv1x1(mid_ch[self.fnum1-1-i],mid_ch[self.fnum1-1-i])for i in range(self.fnum2)])
        self.Cam = nn.ModuleList([CAM(mid_ch[self.fnum1-1-i],mid_ch[self.fnum1-i-2],mid_ch[self.fnum1-i-2],r=r)for i in range(self.fnum2)])

        self.adf = ADF(sum(mid_ch[0:self.fnum1]),out_ch)

    def forward(self,features):
        if(len(features)!=self.fnum1+self.fnum2):
            raise KeyError('Input features dim is wrong')
        hx = [self.reduct[i](features[i]) for i in range(self.fnum1+self.fnum2)]
        hxfused = []
        hxfused.append(hx[self.fnum1-1])
        for i in range(self.fnum2):
            Halfx = self.Gate[i](hx[self.fnum1+self.fnum2-1-i])
            upx = self.MidConv[i](hxfused[i])
            fusex = upx * (upx+Halfx)
            fsx = self.Cam[i](fusex,hx[self.fnum1-i-2])
            hxfused.append(fsx)

        x_out = self.adf(torch.cat(hxfused,dim=1))

        return x_out


class Mutual_AdaptiveFuse_wog(nn.Module):
    def __init__(self,fnum:list, in_ch:list, mid_ch:list, out_ch:int,r=2):
        super(Mutual_AdaptiveFuse_wog,self).__init__()
        self.fnum1 = fnum[0]
        self.fnum2 = fnum[1]
        if(self.fnum1!=self.fnum2+1):
            # raw image level should be 1 larger than 1/2 level
            # so that they can mustually interact with each other
            raise KeyError('The parameter \'fnum\' is wrong')
        if(len(in_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'in_ch\' is wrong')
        if(len(mid_ch)!=self.fnum1+self.fnum2):
            raise KeyError('The parameter \'mid_ch\' is wrong')
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch


        self.reduct = nn.ModuleList([Conv1x1(in_ch[i],mid_ch[i])for i in range(self.fnum1+self.fnum2)])
        self.Gate = nn.ModuleList([Conv3x3(mid_ch[self.fnum1+self.fnum2-1-i],mid_ch[self.fnum1-1-i])for i in range(self.fnum2)])
        self.MidConv = nn.ModuleList([Conv1x1(mid_ch[self.fnum1-1-i],mid_ch[self.fnum1-1-i])for i in range(self.fnum2)])
        self.Cam = nn.ModuleList([CAM(mid_ch[self.fnum1-1-i],mid_ch[self.fnum1-i-2],mid_ch[self.fnum1-i-2],r=r)for i in range(self.fnum2)])

        self.adf = ADF(sum(mid_ch[0:self.fnum1]),out_ch)

    def forward(self,features):
        if(len(features)!=self.fnum1+self.fnum2):
            raise KeyError('Input features dim is wrong')
        hx = [self.reduct[i](features[i]) for i in range(self.fnum1+self.fnum2)]
        hxfused = []
        hxfused.append(hx[self.fnum1-1])
        for i in range(self.fnum2):
            Halfx = self.Gate[i](hx[self.fnum1+self.fnum2-1-i])
            upx = self.MidConv[i](hxfused[i])
            fusex = upx* (upx+Halfx)
            fsx = self.Cam[i](fusex,hx[self.fnum1-i-2])
            hxfused.append(fsx)

        x_out = self.adf(torch.cat(hxfused,dim=1))

        return x_out

class Detail_fusion(nn.Module):
    def __init__(self,in_ch):
        super(Detail_fusion,self).__init__()
        self.in_ch = in_ch
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.detail_weight = Conv1x1(in_ch*2,in_ch)
        self.add_weight = Conv1x1(in_ch,in_ch)

    def forward(self,x,detail):
        hx = x
        maxx = self.maxpool(hx)
        avgx = self.avgpool(hx)
        dweight = self.detail_weight(torch.cat([maxx,avgx],dim = 1))
        addtion = self.add_weight(dweight*detail)
        outx = self.add_weight(hx + addtion)

        return outx



class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )##512

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )#512

        ##############add spatial attention ###Cross UtU############
        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            # nn.Sigmoid(),

            SpatialAttention(kernel_size=3),
            # nn.Conv2d(self.bottleneck_channels, 2, 3, 1, 0),
            # nn.Conv2d(2, 1, 1, 1, 0),
            #nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )#512

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl * topdown_wei)
        xs1 = 2 * xl * topdown_wei  #1
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei    #1
        out2 = self.post(xs2)
        return out1,out2

        ##############################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x
