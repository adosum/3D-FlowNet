import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .util import predict_flow, crop_like, conv_s, conv, deconv
from .util import  conv_3d,deconv_3d
import sys

__all__ = ['flownets_3d']

    
    

    
    

    
class FlowNetS_3d(nn.Module):
    expansion = 1
    def __init__(self,batchNorm=True):
        super(FlowNetS_3d,self).__init__()
        self.batchNorm = batchNorm


        self.conv1   = conv_3d(self.batchNorm,   4,   64, kernel_size=3, stride=2,padding=(2,1,1)) # 4
        self.conv2   = conv_3d(self.batchNorm,  64,  128, kernel_size=3, stride=2,padding=(2,1,1)) # 3
        self.conv3   = conv_3d(self.batchNorm, 128,  256, kernel_size=3, stride=2,padding=(0,1,1)) # 1
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2)
        
        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512,128)
        self.deconv2 = deconv(self.batchNorm, 384+2,64)
        self.deconv1 = deconv(self.batchNorm, 128*3+2+64,4)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.pre_predict4 = conv(self.batchNorm, 256, 32, kernel_size=3, stride=1)
        self.pre_predict3 = conv(self.batchNorm, 256, 32, kernel_size=3, stride=1)
        self.pre_predict2 = conv(self.batchNorm, 128, 32, kernel_size=3, stride=1)
        self.pre_predict1 = conv(self.batchNorm, 128, 32, kernel_size=3, stride=1)


        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384+2, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=128*3+2+64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=64*4+2+4, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)

      
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n) 
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, input):

        
    
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3 = out_conv3.squeeze(2)
        out_conv4 = self.conv4(out_conv3)

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.pre_predict4(self.upsampled_flow4_to_3(out_rconv22)))
        flow4_up = crop_like(flow4, out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3 = self.predict_flow3(self.pre_predict3(self.upsampled_flow3_to_2(concat3)))
        out_conv2_flat = torch.flatten(out_conv2,start_dim=1,end_dim=2)
        flow3_up = crop_like(flow3, out_conv2_flat)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2_flat)

        concat2 = torch.cat((out_conv2_flat,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(self.pre_predict2(self.upsampled_flow2_to_1(concat2)))
        out_conv1_flat = torch.flatten(out_conv1,start_dim=1,end_dim=2)
        flow2_up = crop_like(flow2, out_conv1_flat)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1_flat)

        concat1 = torch.cat((out_conv1_flat,out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(self.pre_predict1(self.upsampled_flow1_to_0(concat1)))
        
   
        
        if self.training:
            return flow1,flow2,flow3,flow4
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets_3d(data=None):
    model = FlowNetS_3d(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

