#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:02:12 2021

@author: user
"""
import argparse
import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
# import torch
from event_utils import *
from objectives import *
from warps import *

if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        gt Ground truth optic flow for event slice
        img_size The size of the event camera sensor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",default='../slider_depth.h5')
    parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))
    args = parser.parse_args()
    B = 10
    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts-ts[0]
    gt_params = tuple(args.gt)
    img_size=tuple(args.img_size)

    start_idx = 20000
    end_idx=start_idx+15000
    blur = None
    img = events_to_image(xs[start_idx:end_idx], ys[start_idx:end_idx], ps[start_idx:end_idx])
    plot_image(img)
    
    # devided the timestamp into [0,B-1] 
    ts_tmp = ts[start_idx:end_idx]
    # ts_tmp = ts_tmp - ts_tmp[0]
    ts_tmp = ts_tmp*(10-1)/ts_tmp[-1]
    