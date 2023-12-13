import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import *


# 图像 multilevel：Raw(RSU*6) + Half-maxpool(尺寸为1/2,RSU*5)


class con_up_fusion3(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(con_up_fusion3, self).__init__()
        self.confuse = nn.Sequential(
        nn.Conv2d(in_ch,out_ch,1,padding=0),
        nn.BatchNorm2d(out_ch),
        nn.Sigmoid()
    )

    def forward(self, con):
        y_con = torch.mean(con, dim=2, keepdim=True)  # (bs,channel,1,W)
        x_con = torch.mean(con, dim=3, keepdim=True)  # (bs,channel,H,1)
        mul_con = torch.matmul(x_con, y_con)  # (bs,channel,H,W)
        ch_con_max = torch.max(con, dim=1, keepdim=True)  # (bs,1,H,W)
        conw = self.confuse(mul_con * ch_con_max[0])
        out = conw * con

        return out


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        # 10 10
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        # 20 20
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        # 40 40
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 80 80
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 160 160
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 320 320
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin, hx2d, hx3d, hx4d, hx5d, hx6d


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        # 10 10
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        # 20 20
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 40 40
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 80 80
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 160 160
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin, hx2d, hx3d, hx4d, hx5d


### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin, hx2d, hx3d, hx4d

    ### RSU-4 ###


class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin, hx2d, hx3d

    ### RSU-4F ###


class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin, hx2d

    ##### NEW-net ####


class MFSRNet(nn.Module):
    # image multilevel ADF block and triple modality fusion in decoder
    # no 1/2 image into connection: con = fuse(raw,1/2) ---> UCF
    # connection and ADF fused in UCF while UP is concate with UCF
    # AdaptiveFuse2
    def __init__(self, in_ch=3, out_ch=1):
        super(MFSRNet, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # level2: 1/2 image
        self.stage7 = RSU6(in_ch, 32, 64)
        self.pool78 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage8 = RSU5(64, 32, 128)
        self.pool89 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage9 = RSU4(128, 64, 256)
        self.pool910 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage10 = RSU4F(256, 128, 256)
        self.pool1011 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage11 = RSU4F(256, 128, 256)

        # decoder
        self.con5 = con_up_fusion3(512, 512)
        self.con4 = con_up_fusion3(512, 512)
        self.con3 = con_up_fusion3(256, 256)
        self.con2 = con_up_fusion3(128, 128)
        self.con1 = con_up_fusion3(64, 64)

        self.con_upd5 = Conv3x3(512, 512)
        self.con_upd4 = Conv3x3(512, 512)
        self.con_upd3 = Conv3x3(256, 256)
        self.con_upd2 = Conv3x3(128, 128)
        self.con_upd1 = Conv3x3(64, 64)

        self.ADF_cup_fuse6 = Conv1x1(1024, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # v1.0
        self.adfuse1 = self.Adaptive_fuse(fnum=1, in_ch=[64], mid_ch=[64], out_ch=64)  # 64,64
        self.adfuse2 = self.Adaptive_fuse(fnum=3, in_ch=[32, 128, 64], mid_ch=[16, 64, 32], out_ch=128)  # 160,128
        self.adfuse3 = self.Adaptive_fuse(fnum=5, in_ch=[32, 32, 256, 32, 128], mid_ch=[16, 16, 64, 16, 64],
                                          out_ch=256)  # 384,256
        self.adfuse4 = self.Adaptive_fuse(fnum=7, in_ch=[32, 32, 64, 512, 32, 32, 256],
                                          mid_ch=[16, 16, 32, 64, 16, 16, 64], out_ch=512)  # 144,512
        self.adfuse5 = self.Adaptive_fuse(fnum=9, in_ch=[32, 32, 64, 128, 512, 32, 32, 64, 256],
                                          mid_ch=[16, 16, 32, 32, 64, 16, 16, 32, 64], out_ch=512)  # 224,512
        self.adfuse6 = self.Adaptive_fuse(fnum=9, in_ch=[32, 32, 64, 128, 512, 32, 32, 64, 256],
                                          mid_ch=[16, 16, 32, 32, 64, 16, 16, 32, 64], out_ch=512)  # 224,512

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

        self.poola = nn.MaxPool2d(2, 2, ceil_mode=True)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):  # fuse_mode='AsymBi'
        # assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        # if fuse_mode == 'BiLocal':
        #     fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # el
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # elif fuse_mode == 'BiGlobal':
        #     fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            raise KeyError()
        return fuse_layer

    def Adaptive_fuse(self, fnum, in_ch, mid_ch, out_ch):
        adfuse = AdaptiveFuse2(fnum=fnum, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch)
        return adfuse

    def forward(self, x):
        
        # stage 1
        hx11, hx12, hx13, hx14, hx15, hx16 = self.stage1(x)

        # stage 2
        hx22, hx23, hx24, hx25, hx26 = self.stage2(self.pool12(hx11))

        # stage 3
        hx33, hx34, hx35, hx36 = self.stage3(self.pool23(hx22))

        # stage 4
        hx44, hx45, hx46 = self.stage4(self.pool34(hx33))

        # stage 5
        hx55, _ = self.stage5(self.pool45(hx44))

        # stage 6
        hx66, _ = self.stage6(self.pool56(hx55))

        # stage 7
        hx72, hx73, hx74, hx75, hx76 = self.stage7(self.poola(x))

        # stage 8
        hx83, hx84, hx85, hx86 = self.stage8(self.pool78(hx72))

        # stage 9
        hx94, hx95, hx96 = self.stage9(self.pool89(hx83))

        # stage 10
        hx105, _ = self.stage10(self.pool910(hx94))

        # stage 11
        hx116, _ = self.stage11(self.pool1011(hx105))

        # ADF block
        hx1 = self.adfuse1([hx11])
        hx2 = self.adfuse2([hx12, hx22, hx72])
        hx3 = self.adfuse3([hx13, hx23, hx33, hx73, hx83])
        hx4 = self.adfuse4([hx14, hx24, hx34, hx44, hx74, hx84, hx94])
        hx5 = self.adfuse5([hx15, hx25, hx35, hx45, hx55, hx75, hx85, hx95, hx105])
        hx6 = self.adfuse6([hx16, hx26, hx36, hx46, hx66, hx76, hx86, hx96, hx116])  # 512

        
        hx6d = self.ADF_cup_fuse6(torch.cat((hx66, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)  # 512 + 512 = 1024 --> 512
        # -------------------- decoder --------------------

        # fusec51,fusec52 = self.fuse5(hx6up, hx5)
        # hx5d = self.stage5d(torch.cat((fusec51, fusec52),1))[0]
        con_up5 = self.con_upd5(self.con5(hx55)+hx5)  # 512
        hx5d = self.stage5d(torch.cat((con_up5, hx6dup), 1))[0]  # 512 + 512 = 1024 --> 512
        hx5dup = _upsample_like(hx5d, hx4)

        # fusec41,fusec42 = self.fuse4(hx5dup, hx4)
        # hx4d = self.stage4d(torch.cat((fusec41,fusec42),1))[0]
        con_up4 = self.con_upd4(self.con4(hx44)+hx4)  # 512
        hx4d = self.stage4d(torch.cat((con_up4, hx5dup), 1))[0]  # 512 + 512 = 1024 --> 256
        hx4dup = _upsample_like(hx4d, hx3)

        # fusec31,fusec32 = self.fuse3(hx4dup, hx3)
        # hx3d = self.stage3d(torch.cat((fusec31,fusec32),1))[0]
        con_up3 = self.con_upd3(self.con3(hx33)+hx3)  # 256
        hx3d = self.stage3d(torch.cat((con_up3, hx4dup), 1))[0]  # 256 + 256 --> 128
        hx3dup = _upsample_like(hx3d, hx2)

        # fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        # hx2d = self.stage2d(torch.cat((fusec21, fusec22), 1))[0]
        con_up2 = self.con_upd2(self.con2(hx22)+hx2)  # 128
        hx2d = self.stage2d(torch.cat((con_up2, hx3dup), 1))[0]  # 128 + 128 --> 64
        hx2dup = _upsample_like(hx2d, hx1)

        
        con_up1 = self.con_upd1(self.con1(hx11)+hx1)  # 64
        hx1d = self.stage1d(torch.cat((con_up1, hx2dup), 1))[0]  # 64 + 64 --> 64

        # side output
        d1 = self.side1(hx1d)

        d22 = self.side2(hx2d)
        d2 = _upsample_like(d22, d1)

        d32 = self.side3(hx3d)
        d3 = _upsample_like(d32, d1)

        d42 = self.side4(hx4d)
        d4 = _upsample_like(d42, d1)

        d52 = self.side5(hx5d)
        d5 = _upsample_like(d52, d1)

        d62 = self.side6(hx6d)
        d6 = _upsample_like(d62, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)#, vis
