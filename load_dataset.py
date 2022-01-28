#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:40:32 2021

@author: user
"""
from os.path import splitext
from os import listdir
import numpy as np
# from glob import glob
import torch
from torch.utils.data import Dataset
# import logging
from PIL import Image
import torchvision.transforms as transforms
import random
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset, get_hot_event_mask, save_image
from data_augmentation import Compose


class TrainDataset(Dataset):
    def __init__(self, event_dir,imgs_dir, dir_gt,data_split,image_size,vox_transforms_list,offset):
        self.imgs_dir = imgs_dir
        self.gt_dir = dir_gt
        self.event_dir = event_dir
        self.image_size = image_size
        self.data_split = data_split
        self.ids = [splitext(file)[0] for file in sorted(listdir(event_dir))
                    if not file.startswith('.')]
        
        self.ids_gray = [splitext(file)[0] for file in sorted(listdir(imgs_dir))
                    if not file.startswith('.')]
        
        self.length = len(self.ids_gray)
        
        
        self.vox_transform = Compose(vox_transforms_list)
        self.combined_voxel_channels = False
        self.sensor_resolution = (260,346)
        self.offset = offset
    def __len__(self):
        return len(self.ids_gray)
    
    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.data_split, *self.sensor_resolution)
        else:
            size = (2*self.data_split, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)
    
    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel
    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.data_split,\
                                               sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.data_split,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0) #voxel_pos, voxel_neg

        # voxel_grid = voxel_grid*self.hot_events_mask 

        return voxel_grid
    def process_event(self,events,former_gray,latter_gray,flow):
        xs = events[:,0]
        ys = events[:,1]
        ts = events[:,2]
        ps = events[:,3]
        try:
            ts_0, ts_k  = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))
        
        voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        seed = random.randint(0, 2 ** 32)
        voxel = self.transform_voxel(voxel, seed).float()
        
        
        former_gray = former_gray.unsqueeze(0)
        latter_gray = latter_gray.unsqueeze(0)
        
        former_gray = self.transform_voxel(former_gray, seed)
        latter_gray = self.transform_voxel(latter_gray, seed)
        flow = flow.permute(2,0,1)
        dt = ts_k - ts_0
        # flow = flow * dt
        flow = self.transform_voxel(flow, seed)
        
        
        return voxel,former_gray,latter_gray,torch.tensor(dt, dtype=torch.float64),flow
            
    def __getitem__(self, i):
        if i + self.offset < self.length and i > self.offset:
            img = np.load(self.imgs_dir +'/'+ str(int(i))+'.npy')
            # flo =  np.load(self.gt_dir +'/'+ str(int(i))+'.npy')
            eventdata_raw =  np.load(self.event_dir +'/'+ str(int(i+1))+'.npy')
            # next_img =   Image.fromarray(np.load(self.imgs_dir +'/'+ str(int(i+1))+'.npy'))
            next_img =   np.load(self.imgs_dir +'/'+ str(int(i+1))+'.npy')
            flow = np.load(self.gt_dir + '/' + str(int(i))+'.npy') 
            # img = self.preprocess(img, self.image_size)
            # mask = self.preprocess(mask, self.image_size)
            # mask = np.where(mask == 1, 1, 0)
            # next_img = self.preprocess(next_img, self.image_size)
            voxel,img,next_img,delta_t,flow = self.process_event(eventdata_raw,\
                                                                     torch.tensor(img),\
                                                                torch.tensor(next_img),torch.tensor(flow))
            # voxel_b = 
            eventdata = torch.zeros((4,int(self.data_split/2),self.image_size, self.image_size),\
                                    dtype=torch.float)  
            event_data_b = torch.zeros((4,int(self.data_split/2),self.image_size, self.image_size),\
                                    dtype=torch.float)
            eventdata[0,...] = voxel[0:5,...]
            eventdata[1,...] = voxel[5:10,...]
            eventdata[2,...] = voxel[10:15,...]
            eventdata[3,...] = voxel[15:20,...]
            event_data_b[1,...] = voxel[0:5,...]
            event_data_b[0,...] = voxel[5:10,...]
            event_data_b[3,...] = voxel[10:15,...]
            event_data_b[2,...] = voxel[15:20,...]
            # event_data_b = torch.flip(eventdata,[0])
            # img = img.permute((2, 0, 1))
            # next_img = next_img.permute((2, 0, 1))
            # torch.nn.functional.interpolate(flo,(self.image_size, self.image_size),
            #                                 mode='bilinear', align_corners=False)
            # torch.nn.functional.interpolate(img,(self.image_size, self.image_size),
            #                                 mode='bilinear', align_corners=False)
            # torch.nn.functional.interpolate(next_img,(self.image_size, self.image_size),
            #                                 mode='bilinear', align_corners=False)
            assert img.shape[1] == self.image_size, \
                f'The image shape should be {self.image_size}, ' \
                f'but loaded images have {img.shape[1]}. Please check that ' \
                'the images are loaded correctly.'
            assert eventdata.shape[2] == self.image_size, \
                f'The eventdata shape should be {self.image_size}, ' \
                f'but loaded eventdata have {img.shape[1]}. Please check that ' \
                'the images are loaded correctly.'
            assert flow.shape[1] == self.image_size, 'The flow shape is wrong'

                
            return {
            'former_image': img.float(),
            'eventdata':eventdata.float(),
            'eventdata_b':event_data_b,
            # 'mask': mask,
            'latter_image':next_img.float(),
            # 'eventdata_raw':torch.FloatTensor(event_image)
            # 'next_eventframe':next_eventframe
            'flow':flow.float(),
            'delta_t':delta_t,
            }
        else:
            
            return {
                'former_image': torch.zeros((1, self.image_size, self.image_size), dtype=torch.float),
                'eventdata':torch.zeros((4,int(self.data_split/2),self.image_size, self.image_size),\
                                        dtype=torch.float),
                'eventdata_b':torch.zeros((4,int(self.data_split/2),self.image_size, self.image_size),\
                                        dtype=torch.float),
                # 'mask': mask,
                'latter_image':torch.zeros((1, self.image_size, self.image_size), dtype=torch.float),
                # 'eventdata_raw':torch.FloatTensor(event_image)
                # 'next_eventframe':next_eventframe
                'flow':torch.zeros((2, self.image_size, self.image_size), dtype=torch.float),
                'delta_t':torch.tensor(0., dtype=torch.float64),
                }