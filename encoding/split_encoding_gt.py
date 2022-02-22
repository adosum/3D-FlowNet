#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:17:58 2021

@author: user
"""
import numpy as np
import os
import h5py
import argparse
from multiscaleloss import estimate_corresponding_gt_flow


parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS', help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='../datasets/indoor_flying1/indoor_flying1_data.hdf5', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
parser.add_argument('--gt-path', type=str, default='../datasets/indoor_flying1/indoor_flying1_gt.hdf5', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()


save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
  os.makedirs(save_path)
  
gt_dir = os.path.join(save_path, 'gt_data')
if not os.path.exists(gt_dir):
  os.makedirs(gt_dir)
  

class Events(object):
    def __init__(self, width=346, height=260):
        # self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)], shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self,gt_temp=0,gt_ts_temp=0, image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0):
        print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)

        split_interval = image_raw_ts_temp.shape[0]

        

        t_index = 0
        U_gt_all = np.array(gt_temp[:, 0, :, :])
        V_gt_all = np.array(gt_temp[:, 1, :, :])
        for i in range(split_interval-dt_time_temp-2):
            
            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, 
                                                        gt_ts_temp, 
                                                        np.array(image_raw_ts[i]), 
                                                        np.array(image_raw_ts[i+dt_time_temp]))
            gt_flow = np.stack((U_gt, V_gt), axis=2)
            t_index = t_index + 1

            np.save(os.path.join(gt_dir, str(i)), gt_flow)
            # np.save(os.path.join(mask_dir, str(i)), mask[i,:,:])



d_set = h5py.File(args.data_path, 'r')

# raw_data = d_set['events']
# image_raw_event_inds = d_set['image_raw_event_inds']
# image_raw_ts = np.float64(d_set['image_raw_ts'])
# gray_image = d_set['image_raw']
# mask = d_set['mask']
# raw_data = d_set['davis']['left']['events']
image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
# gray_image = d_set['davis']['left']['image_raw']

d_set = None

d_label = h5py.File(args.gt_path, 'r')
gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
d_label = None

dt_time = 1

td = Events()
# Events
# td.generate_fimage(input_event=raw_data, gray=gray_image, mask = mask,
#                    image_raw_event_inds_temp=image_raw_event_inds, 
#                    image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
# td.generate_fimage(input_event=raw_data, gray=gray_image, 
#                    image_raw_event_inds_temp=image_raw_event_inds, 
#                    image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)

td.generate_fimage(gt_temp=gt_temp,gt_ts_temp=gt_ts_temp,
                   image_raw_event_inds_temp=image_raw_event_inds, 
                   image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)

raw_data = None


print('Encoding complete!')
