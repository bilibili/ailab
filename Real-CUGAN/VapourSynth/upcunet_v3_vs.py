import torch
from torch import nn as nn
from torch.nn import functional as F
import os,sys
import numpy as np
root_path=os.path.abspath('.')
sys.path.append(root_path)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x,x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x
class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z
class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1,x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1,x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):#conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2,x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1,x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z
class UpCunet2x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x,tile_mode):#1.7G
        n, c, h0, w0 = x.shape
        if(tile_mode==0):#不tile
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x1)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
            return x
        elif(tile_mode==1):# 对长边减半
            if(w0>=h0):
                crop_size_w=((w0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_h=(h0-1)//2*2+2#能被2整除
            else:
                crop_size_h=((h0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_w=(w0-1)//2*2+2#能被2整除
            crop_size=(crop_size_h,crop_size_w)#6.6G
        elif(tile_mode==2):#hw都减半
            crop_size=(((h0-1)//4*4+4)//2,((w0-1)//4*4+4)//2)#5.6G
        elif(tile_mode==3):#hw都三分之一
            crop_size=(((h0-1)//6*6+6)//3,((w0-1)//6*6+6)//3)#4.2G
        elif(tile_mode==4):#hw都四分之一
            crop_size=(((h0-1)//8*8+8)//4,((w0-1)//8*8+8)//4)#3.7G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x=F.pad(x,(18,18+pw-w0,18,18+ph-h0),'reflect')
        n,c,h,w=x.shape
        se_mean0=torch.zeros((n,64,1,1)).to(x.device)
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        n_patch=0
        tmp_dict={}
        opt_res_dict={}
        for i in range(0,h-36,crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-36,crop_size[1]):
                x_crop=x[:,:,i:i+crop_size[0]+36,j:j+crop_size[1]+36]
                n,c1,h1,w1=x_crop.shape
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 128, 1, 1)).to(x.device)#64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop,se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        se_mean0 = torch.zeros((n, 128, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x3=self.unet2.forward_b(tmp_x2)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 64, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        for i in range(0,h-36,crop_size[0]):
            opt_res_dict[i]={}
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                x1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                x_crop = torch.add(x0, x1)#x0是unet2的最终输出
                opt_res_dict[i][j] = x_crop
        del tmp_dict
        torch.cuda.empty_cache()
        res = torch.zeros((n, c, h * 2 - 72, w * 2 - 72)).to(x.device)
        if ("Half" in x.type()):
            res=res.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                res[:, :, i * 2:i * 2 + h1 * 2 - 72, j * 2:j * 2 + w1 * 2 - 72]=opt_res_dict[i][j]
        del opt_res_dict
        torch.cuda.empty_cache()
        if(w0!=pw or h0!=ph):res=res[:,:,:h0*2,:w0*2]
        return res#
class UpCunet3x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet3x, self).__init__()
        self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x,tile_mode):#1.7G
        n, c, h0, w0 = x.shape
        if(tile_mode==0):#不tile
            ph = ((h0 - 1) // 4 + 1) * 4
            pw = ((w0 - 1) // 4 + 1) * 4
            x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x1)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 3, :w0 * 3]
            return x
        elif(tile_mode==1):# 对长边减半
            if(w0>=h0):
                crop_size_w=((w0-1)//8*8+8)//2#减半后能被4整除，所以要先被8整除
                crop_size_h=(h0-1)//4*4+4#能被4整除
            else:
                crop_size_h=((h0-1)//8*8+8)//2#减半后能被4整除，所以要先被8整除
                crop_size_w=(w0-1)//4*4+4#能被4整除
            crop_size=(crop_size_h,crop_size_w)#6.6G
        elif(tile_mode==2):#hw都减半
            crop_size=(((h0-1)//8*8+8)//2,((w0-1)//8*8+8)//2)#5.6G
        elif(tile_mode==3):#hw都三分之一
            crop_size=(((h0-1)//12*12+12)//3,((w0-1)//12*12+12)//3)#4.2G
        elif(tile_mode==4):#hw都四分之一
            crop_size=(((h0-1)//16*16+16)//4,((w0-1)//16*16+16)//4)#3.7G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x=F.pad(x,(14,14+pw-w0,14,14+ph-h0),'reflect')
        n,c,h,w=x.shape
        se_mean0=torch.zeros((n,64,1,1)).to(x.device)
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        n_patch=0
        tmp_dict={}
        opt_res_dict={}
        for i in range(0,h-28,crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-28,crop_size[1]):
                x_crop=x[:,:,i:i+crop_size[0]+28,j:j+crop_size[1]+28]
                n,c1,h1,w1=x_crop.shape
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 128, 1, 1)).to(x.device)#64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-28,crop_size[0]):
            for j in range(0,w-28,crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop,se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        se_mean0 = torch.zeros((n, 128, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        for i in range(0,h-28,crop_size[0]):
            for j in range(0,w-28,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x3=self.unet2.forward_b(tmp_x2)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 64, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-28,crop_size[0]):
            for j in range(0,w-28,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        for i in range(0,h-28,crop_size[0]):
            opt_res_dict[i]={}
            for j in range(0,w-28,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                x1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                x_crop = torch.add(x0, x1)#x0是unet2的最终输出
                opt_res_dict[i][j]=x_crop#
        del tmp_dict
        torch.cuda.empty_cache()
        res = torch.zeros((n, c, h * 3 - 84, w * 3 - 84)).to(x.device)
        if ("Half" in x.type()):
            res=res.half()
        for i in range(0,h-28,crop_size[0]):
            for j in range(0,w-28,crop_size[1]):
                res[:, :, i*3:i*3 + h1*3-84, j*3:j*3 + w1*3-84] =opt_res_dict[i][j]
        del opt_res_dict
        torch.cuda.empty_cache()
        if(w0!=pw or h0!=ph):res=res[:,:,:h0*3,:w0*3]
        return res
class UpCunet4x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet4x, self).__init__()
        self.unet1 = UNet1(in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps=nn.PixelShuffle(2)
        self.conv_final=nn.Conv2d(64,12,3,1,padding=0,bias=True)
    def forward(self, x,tile_mode):
        n, c, h0, w0 = x.shape
        x00 = x
        if(tile_mode==0):#不tile
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            x = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x1)
            x=self.conv_final(x)
            x=F.pad(x,(-1,-1,-1,-1))
            x=self.ps(x)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 4, :w0 * 4]
            x+=F.interpolate(x00, scale_factor=4, mode='nearest')
            return x
        elif(tile_mode==1):# 对长边减半
            if(w0>=h0):
                crop_size_w=((w0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_h=(h0-1)//2*2+2#能被2整除
            else:
                crop_size_h=((h0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_w=(w0-1)//2*2+2#能被2整除
            crop_size=(crop_size_h,crop_size_w)#6.6G
        elif(tile_mode==2):#hw都减半
            crop_size=(((h0-1)//4*4+4)//2,((w0-1)//4*4+4)//2)#5.6G
        elif(tile_mode==3):#hw都三分之一
            crop_size=(((h0-1)//6*6+6)//3,((w0-1)//6*6+6)//3)#4.1G
        elif(tile_mode==4):#hw都四分之一
            crop_size=(((h0-1)//8*8+8)//4,((w0-1)//8*8+8)//4)#3.7G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x=F.pad(x,(19,19+pw-w0,19,19+ph-h0),'reflect')
        n,c,h,w=x.shape
        se_mean0=torch.zeros((n,64,1,1)).to(x.device)
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        n_patch=0
        tmp_dict={}
        opt_res_dict={}
        for i in range(0,h-38,crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-38,crop_size[1]):
                x_crop=x[:,:,i:i+crop_size[0]+38,j:j+crop_size[1]+38]
                n,c1,h1,w1=x_crop.shape
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 128, 1, 1)).to(x.device)#64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-38,crop_size[0]):
            for j in range(0,w-38,crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop,se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        se_mean0 = torch.zeros((n, 128, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        for i in range(0,h-38,crop_size[0]):
            for j in range(0,w-38,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x3=self.unet2.forward_b(tmp_x2)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 64, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-38,crop_size[0]):
            for j in range(0,w-38,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        for i in range(0,h-38,crop_size[0]):
            opt_res_dict[i]={}
            for j in range(0,w-38,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                x1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                x_crop = torch.add(x0, x1)#x0是unet2的最终输出
                x_crop=self.conv_final(x_crop)
                x_crop = F.pad(x_crop, (-1, -1, -1, -1))
                x_crop=self.ps(x_crop)
                opt_res_dict[i][j] = x_crop
        del tmp_dict
        torch.cuda.empty_cache()
        res = torch.zeros((n, c, h * 4 - 152, w * 4 - 152)).to(x.device)
        if ("Half" in x.type()):
            res=res.half()
        for i in range(0,h-38,crop_size[0]):
            for j in range(0,w-38,crop_size[1]):
                # print(opt_res_dict[i][j].shape,res[:, :, i * 4:i * 4 + h1 * 4 - 144, j * 4:j * 4 + w1 * 4 - 144].shape)
                res[:, :, i * 4:i * 4 + h1 * 4 - 152, j * 4:j * 4 + w1 * 4 - 152]=opt_res_dict[i][j]
        del opt_res_dict
        torch.cuda.empty_cache()
        if(w0!=pw or h0!=ph):res=res[:,:,:h0*4,:w0*4]
        res += F.interpolate(x00, scale_factor=4, mode='nearest')
        return res#
class pixel_unshuffle(nn.Module):
    def __init__(self, ratio=2):
        super(pixel_unshuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

class UpCunet2x_fast(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x_fast, self).__init__()
        self.unet1 = UNet1(12, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps=nn.PixelShuffle(2)
        self.conv_final=nn.Conv2d(64,12,3,1,padding=0,bias=True)
        self.inv=pixel_unshuffle(2)

    def forward(self, x,tile_mode):
        n, c, h0, w0 = x.shape
        x00 = x
        if(tile_mode==0):#不tile
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            x = F.pad(x, (38, 38 + pw - w0, 38, 38 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.inv(x)  # +18
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x1)
            x=self.conv_final(x)
            # with open(r"C:\Users\liujing\Desktop\log.txt","a+")as f:
            #     f.write("%s"%(str(x.shape)))
            #     f.flush()
            x=F.pad(x,(-1,-1,-1,-1))
            x=self.ps(x)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
            x+=F.interpolate(x00, scale_factor=2, mode='nearest')
            return x
        elif(tile_mode==1):# 对长边减半
            if(w0>=h0):
                crop_size_w=((w0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_h=(h0-1)//2*2+2#能被2整除
            else:
                crop_size_h=((h0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_w=(w0-1)//2*2+2#能被2整除
            crop_size=(crop_size_h,crop_size_w)#6.6G
        elif(tile_mode==2):#hw都减半
            crop_size=(((h0-1)//4*4+4)//2,((w0-1)//4*4+4)//2)#5.6G
        elif(tile_mode==3):#hw都三分之一
            crop_size=(((h0-1)//6*6+6)//3,((w0-1)//6*6+6)//3)#4.1G
        elif(tile_mode==4):#hw都四分之一
            crop_size=(((h0-1)//8*8+8)//4,((w0-1)//8*8+8)//4)#3.7G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x=F.pad(x,(38,38+pw-w0,38,38+ph-h0),'reflect')
        n,c,h,w=x.shape
        se_mean0=torch.zeros((n,64,1,1)).to(x.device)
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        n_patch=0
        tmp_dict={}
        opt_res_dict={}
        for i in range(0,h-76,crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-76,crop_size[1]):
                x_crop=x[:,:,i:i+crop_size[0]+76,j:j+crop_size[1]+76]
                n,c1,h1,w1=x_crop.shape
                x_crop=self.inv(x_crop)
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 128, 1, 1)).to(x.device)#64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-76,crop_size[0]):
            for j in range(0,w-76,crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop,se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        se_mean0 = torch.zeros((n, 128, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        for i in range(0,h-76,crop_size[0]):
            for j in range(0,w-76,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x3=self.unet2.forward_b(tmp_x2)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 64, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-76,crop_size[0]):
            for j in range(0,w-76,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        for i in range(0,h-76,crop_size[0]):
            opt_res_dict[i]={}
            for j in range(0,w-76,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                x1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                x_crop = torch.add(x0, x1)#x0是unet2的最终输出
                x_crop=self.conv_final(x_crop)
                x_crop = F.pad(x_crop, (-1, -1, -1, -1))
                x_crop=self.ps(x_crop)
                opt_res_dict[i][j] = x_crop
        del tmp_dict
        torch.cuda.empty_cache()
        res = torch.zeros((n, c, h * 2 - 152, w * 2 - 152)).to(x.device)
        if ("Half" in x.type()):
            res=res.half()
        for i in range(0,h-76,crop_size[0]):
            for j in range(0,w-76,crop_size[1]):
                # print(opt_res_dict[i][j].shape,res[:, :, i * 4:i * 4 + h1 * 4 - 144, j * 4:j * 4 + w1 * 4 - 144].shape)
                res[:, :, i * 2:i * 2 + h1 * 2 - 152, j * 2:j * 4 + w1 * 2 - 152]=opt_res_dict[i][j]
        del opt_res_dict
        torch.cuda.empty_cache()
        if(w0!=pw or h0!=ph):res=res[:,:,:h0*2,:w0*2]
        res += F.interpolate(x00, scale_factor=2, mode='nearest')
        return res#
class RealWaifuUpScaler(object):
    def __init__(self,scale,weight_path,half,device,tile_mode):
        weight = torch.load(weight_path, map_location="cpu")
        self.model=eval("UpCunet%sx"%scale)()
        # self.model=UpCunet2x_fast()
        if(half==True):self.model=self.model.half().to(device)
        else:self.model=self.model.to(device)
        self.model.load_state_dict(weight, strict=True)
        self.model.eval()
        self.half=half
        self.device=device
        self.scale=scale
        self.tile_mode=tile_mode

    def frame_to_tensor(self,f):
        arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
        t=torch.from_numpy(arr).unsqueeze(0)
        if(self.half==True):t=t.half()
        return t.to(self.device)

    def tensor_to_frame(self,t,f):
        arr = t.squeeze(0).detach().cpu().numpy()
        for plane in range(f.format.num_planes):
            np.copyto(np.asarray(f[plane]), arr[plane, :, :])
        return f

    def inference(self,n,f):
        with torch.no_grad():
            img = self.frame_to_tensor(f[0])
            output = self.model(img, self.tile_mode)
            return self.tensor_to_frame(output, f[1].copy())

    def __call__(self, clip):
        new_clip = clip.std.BlankClip(width=clip.width * self.scale, height=clip.height * self.scale)
        return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=self.inference)

