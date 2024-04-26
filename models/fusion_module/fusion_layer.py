import torch
from torch import nn
import os
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
import numpy as np
import cv2


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class EgF(nn.Module):

    def __init__(self, inplanes, planes, version='SREF'):
        super(EgF, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.version = version

        ## rgb channel attention
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

        ## event channel attention
        self.e_conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.e_softmax = nn.Softmax(dim=2)
        self.e_channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

        self.conv1x1_fusion = nn.Conv2d(self.inplanes*2, self.inplanes, kernel_size=1, bias=False)
        self.conv3x3_fusion = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1_rgb = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        self.conv1_evt = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(nn.Conv2d(self.inplanes*2, self.inplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Sequential(nn.Conv2d(self.inplanes*3, self.inplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))

        
        self.reset_parameters()
        print(f'Fusion Module: {version}')

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_mul_conv)

        kaiming_init(self.e_conv_mask, mode='fan_in')
        self.e_conv_mask.inited = True
        last_zero_init(self.e_channel_mul_conv)


    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context
    
    def e_spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.e_conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.e_softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, depth_feature):
        '''
        x: RGB input
        depth_feature: Event input
        '''
        
        ## Symmetric RGB-Event Fusion (SREF)
        event_context = self.e_spatial_pool(depth_feature)  
        event_channel_mul_term = torch.sigmoid(self.e_channel_mul_conv(event_context))
        fea_e = depth_feature * event_channel_mul_term
        event_out1 = torch.sigmoid(fea_e)
        f_out = x * event_out1 + x

        rgb_context = self.spatial_pool(x)
        rgb_channel_mul_term = torch.sigmoid(self.channel_mul_conv(rgb_context))
        fea_r = x * rgb_channel_mul_term
        rgb_out1 = torch.sigmoid(fea_r)
        e_out = depth_feature * rgb_out1 + depth_feature

        rgb2 = self.conv1_rgb(f_out)
        evt2 = self.conv1_evt(e_out)

        max_rgb = torch.reshape(rgb2,[rgb2.shape[0],1,rgb2.shape[1],rgb2.shape[2],rgb2.shape[3]])
        max_evt = torch.reshape(evt2,[evt2.shape[0],1,evt2.shape[1],evt2.shape[2],evt2.shape[3]])
        max_cat = torch.cat((max_rgb, max_evt), dim=1)

        max_out = max_cat.max(dim=1)[0]
        mean_out = max_cat.mean(dim=1)

        out_max_avg = torch.cat((max_out, mean_out), dim=1)

        out = out_max_avg

        out = self.conv_out(out)

        return out, fea_e


class basic_fusion_module(nn.Module):
    def __init__(self,
                 input_dim,
                 version='SREF',
                 time_step=None):
        super(basic_fusion_module, self).__init__()
        self.version = version
        self.SREF = EgF(inplanes=input_dim, planes=input_dim//16, version=self.version)
        self.conv0_evt_max = nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv0_evt_avg = nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.temporal_att = nn.Sequential(nn.Conv2d(input_dim*2, input_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True))
        
    
    def forward(self, RGB_data, Event_data):
        '''
        Event_data: [N T C H W]  C=256/512/1024
        '''
        ## event temporal embedding

        evt_max = torch.max(Event_data,1)[0]
        evt_avg = torch.mean(Event_data,1)

        evt0_max = self.conv0_evt_max(evt_max)  ## N, C, H, W
        evt0_avg = self.conv0_evt_avg(evt_avg)

        ## Temporal Attention Module 
        temporal_event = torch.cat((evt0_max, evt0_avg), dim=1)  ## N, 2C, H, W
        temporal_event = self.temporal_att(temporal_event)

        output, event_fea = self.SREF(RGB_data, temporal_event)

        return output

