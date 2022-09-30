import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from data.utils import pad_tensor, pad_tensor_back

def get_generator(cfg, device):
    if cfg.generator == "Unet_resize_conv":
        return DataParallel(Unet_resize_conv(cfg)).to(device)
    elif cfg.generator == "Unet_resize_conv_spade":
        return DataParallel(Unet_resize_conv_spade(cfg)).to(device)
    else:
        raise "Unknown Generator"

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, norm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(DoubleConvBlock, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(in_channels, out_channels, padding=padding),
            ConvBlock(out_channels, out_channels, padding=padding)
            )

    def forward(self, input):
        return self.main(input)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        param_free_norm_type = "none"

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'none':
            self.param_free_norm = nn.Identity()
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # normalized = x

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        self.gamma = self.mlp_gamma(actv)
        self.beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + self.gamma) + self.beta

        return out

class ConvSPADEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, has_bn=True, has_spade=True):
        super(ConvSPADEBlock, self).__init__()
        self.has_bn = has_bn
        self.has_spade = has_spade
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.main = nn.Sequential(*layers)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.has_spade:
            self.spade = SPADE(norm_nc=out_channels, label_nc=1)
        

    def forward(self, input):
        x, alpha_map = input
        x = self.main(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_spade:
            x = self.spade(x, alpha_map)
        return x, alpha_map

class DoubleConvSPADEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, has_bn=True, has_spade=True):
        super(DoubleConvSPADEBlock, self).__init__()
        self.main = nn.Sequential(
            ConvSPADEBlock(in_channels, out_channels, padding=padding, has_bn=has_bn, has_spade=has_spade),
            ConvSPADEBlock(out_channels, out_channels, padding=padding, has_bn=has_bn, has_spade=has_spade)
            )

    def forward(self, input):
        return self.main(input)[0]

class Unet_resize_conv(nn.Module):
    def __init__(self, cfg):
        super(Unet_resize_conv, self).__init__()

        p = 1

        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)

        self.dcb1 = DoubleConvBlock(4, 32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.dcb2 = DoubleConvBlock(32, 64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.dcb3 = DoubleConvBlock(64, 128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.dcb4 = DoubleConvBlock(128, 256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.cb5_1 = ConvBlock(256, 512)
        self.cb5_2 = ConvBlock(512, 512)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.dcb6 = DoubleConvBlock(512, 256)


        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.dcb7 = DoubleConvBlock(256, 128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.dcb8 = DoubleConvBlock(128, 64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.cb9_1 = ConvBlock(64, 32)
        self.cb9_2 = ConvBlock(32, 32, norm=False)

        self.conv10 = nn.Conv2d(32, 3, 1)


    def forward(self, input, gray):
        downsampled_input = False
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            downsampled_input = True
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)

        gray_2 = self.downsample_1(gray)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
        gray_5 = self.downsample_4(gray_4)

        conv1 = self.dcb1(torch.cat((input, gray), 1))        
        x = self.max_pool1(conv1)

        conv2 = self.dcb2(x)
        x = self.max_pool2(conv2)

        conv3 = self.dcb3(x)
        x = self.max_pool3(conv3)

        conv4 = self.dcb4(x)
        x = self.max_pool4(conv4)

        x = self.cb5_1(x)
        x = x*gray_5
        conv5 = self.cb5_2(x)            


        conv5 = F.interpolate(conv5, scale_factor=2, mode='bilinear')
        conv4 = conv4*gray_4 
        x = torch.cat([self.deconv5(conv5), conv4], 1)
        conv6 = self.dcb6(x)

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear')
        conv3 = conv3*gray_3
        x = torch.cat([self.deconv6(conv6), conv3], 1)
        conv7 = self.dcb7(x)

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear')
        conv2 = conv2*gray_2
        x = torch.cat([self.deconv7(conv7), conv2], 1)
        conv8 = self.dcb8(x)

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear')
        conv1 = conv1*gray
        x = torch.cat([self.deconv8(conv8), conv1], 1)
        conv9 = self.cb9_2(self.cb9_1(x))

        latent = self.conv10(conv9)

        latent = latent*gray

        output = latent + input

            
                
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        
        if downsampled_input:
            output = F.interpolate(output, scale_factor=2, mode='bilinear')
            gray = F.interpolate(gray, scale_factor=2, mode='bilinear')

        return output, [latent]*8

class Unet_resize_conv_spade(nn.Module):
    def __init__(self, cfg):
        super(Unet_resize_conv_spade, self).__init__()
        p = 1

        self.dcb1 = DoubleConvSPADEBlock(3, 32, has_bn=cfg.has_bn)
        self.max_pool1 = nn.MaxPool2d(2)

        self.dcb2 = DoubleConvSPADEBlock(32, 64, has_bn=cfg.has_bn)
        self.max_pool2 = nn.MaxPool2d(2)

        self.dcb3 = DoubleConvSPADEBlock(64, 128, has_bn=cfg.has_bn)
        self.max_pool3 = nn.MaxPool2d(2)

        self.dcb4 = DoubleConvSPADEBlock(128, 256, has_bn=cfg.has_bn)
        self.max_pool4 = nn.MaxPool2d(2)

        self.cb5_1 = ConvSPADEBlock(256, 512, has_bn=cfg.has_bn)
        self.cb5_2 = ConvSPADEBlock(512, 512, has_bn=cfg.has_bn)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.dcb6 = DoubleConvSPADEBlock(512, 256, has_bn=cfg.has_bn)


        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.dcb7 = DoubleConvSPADEBlock(256, 128, has_bn=cfg.has_bn)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.dcb8 = DoubleConvSPADEBlock(128, 64, has_bn=cfg.has_bn)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.cb9_1 = ConvSPADEBlock(64, 32, has_bn=cfg.has_bn)
        self.cb9_2 = ConvSPADEBlock(32, 32, has_bn=False, has_spade=False)

        self.conv10 = nn.Conv2d(32, 3, 1)


    def forward(self, input, alpha_map):
        downsampled_input = False
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            downsampled_input = True
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        conv1 = self.dcb1([input, alpha_map])        
        x = self.max_pool1(conv1)

        conv2 = self.dcb2([x, alpha_map])
        x = self.max_pool2(conv2)

        conv3 = self.dcb3([x, alpha_map])
        x = self.max_pool3(conv3)

        conv4 = self.dcb4([x, alpha_map])
        x = self.max_pool4(conv4)

        x = self.cb5_1([x, alpha_map])[0]
        conv5 = self.cb5_2([x, alpha_map])[0]


        conv5 = F.interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([self.deconv5(conv5), conv4], 1)
        conv6 = self.dcb6([x, alpha_map])

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([self.deconv6(conv6), conv3], 1)
        conv7 = self.dcb7([x, alpha_map])

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([self.deconv7(conv7), conv2], 1)
        conv8 = self.dcb8([x, alpha_map])

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([self.deconv8(conv8), conv1], 1)
        conv9_1 = self.cb9_1([x, alpha_map])
        conv9, _ = self.cb9_2(conv9_1)

        latent = self.conv10(conv9)


        output = latent + input

            
                
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        
        if downsampled_input:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output, latent
