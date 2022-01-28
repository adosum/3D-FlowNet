#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:31:15 2021

@author: user
"""
import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import datasets
from multiscaleloss import estimate_corresponding_gt_flow, flow_error_dense
import datetime
from torch.utils.tensorboard import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint
import cv2
import torch
import os, os.path
from load_dataset import TrainDataset
from torch.utils.data import DataLoader
import models
import numpy as np
import h5py
# import random
from vis_utils import *
# from PIL import Image
from data_augmentation import RandomCrop,RandomFlip,CenterCrop

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='3D-FlowNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='datasets',
                    help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='3d_flownets',
                    help='results save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets_3d',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('--loss', default='photometric',choices=['sos','soe','photometric','sosa','r1','r2'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=800, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[1, 1, 1, 1, 1, 1], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--evaluate-interval', default=5, type=int, metavar='N',
                    help='Evaluate every \'evaluate interval\' epochs ')
parser.add_argument('--print-freq', '-p', default=8000, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--div-flow', default=1,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[5,10,15,20,25,30,35,40,45,50,55,70,90], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--render', dest='render', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_resize = 256
data_split = 10
trainenv = 'outdoor_day2'
testenv = 'indoor_flying1'

traindir = os.path.join(args.data, trainenv)
testdir = os.path.join(args.data, testenv)

trainfile = traindir + '/' + trainenv + '_data.hdf5'
testfile = testdir + '/' + testenv + '_data.hdf5'


def validate(test_loader, model, epoch, output_writers):
    global args, image_resize
    

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    AEE_sum = 0.
    AEE_sum_sum = 0.
    AEE_sum_gt = 0.
    AEE_sum_sum_gt = 0.
    percent_AEE_sum = 0.
    iters = 0.
    scale = 1
    test_loader_iter = iter(test_loader)
 
    for i in range(len(test_loader)):
        data = next(test_loader_iter)
        event_data = data['eventdata']
        former_gray = data['former_image'].squeeze(0).squeeze(0).numpy()
        output = model(event_data)
        gt_flow = data['flow'].squeeze(0).permute(1,2,0).numpy()
        pred_flow = np.zeros((image_resize, image_resize, 2))
        output_temp = output.cpu()
        # resize the output
        output_temp = torch.nn.functional.interpolate(output_temp,(image_resize, image_resize),mode='bilinear', align_corners=False)
        
        pred_flow[:, :, 0] = cv2.resize(np.array(output_temp[0, 0, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)
        pred_flow[:, :, 1] = cv2.resize(np.array(output_temp[0, 1, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)

        if epoch < 0:
            mask_temp = torch.sum(torch.sum(event_data.squeeze(0), 0), 0)
            
            mask_temp_np = np.squeeze(np.array(mask_temp)) > 0
            
           
            if args.render:
                cv2.imshow('event Image', np.array(mask_temp))
                cv2.imwrite('eventimgs/'+str(i)+'.png',np.array(mask_temp)*255)
            
            if args.render:
                cv2.imshow('Gray Image', former_gray/255)
                

            out_temp = np.array(output_temp.cpu().detach())
            x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
            y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
            flow_rgb = flow_viz_np(x_flow, y_flow)
            if args.render:
                cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
                cv2.imwrite('imgs/'+str(i)+'.png',flow_rgb)
            gt_flow_x = cv2.resize(gt_flow[:, :, 0], (scale * image_resize, scale * image_resize),interpolation=cv2.INTER_LINEAR)
            gt_flow_y = cv2.resize(gt_flow[:, :, 1], (scale * image_resize, scale * image_resize),interpolation=cv2.INTER_LINEAR)
            gt_flow_large = flow_viz_np(gt_flow_x, gt_flow_y)
            if args.render:
                cv2.imshow('GT Flow', cv2.cvtColor(gt_flow_large, cv2.COLOR_BGR2RGB))
               
            masked_x_flow = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np), (scale*image_resize,scale* image_resize), interpolation=cv2.INTER_LINEAR)
            masked_y_flow = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np), (scale*image_resize, scale*image_resize), interpolation=cv2.INTER_LINEAR)
            flow_rgb_masked = flow_viz_np(masked_x_flow, masked_y_flow)
            if args.render:
                cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))
                
            gt_flow_cropped = gt_flow
            gt_flow_masked_x = cv2.resize(gt_flow_cropped[:, :, 0]*mask_temp_np, (scale*image_resize, scale*image_resize),interpolation=cv2.INTER_LINEAR)
            gt_flow_masked_y = cv2.resize(gt_flow_cropped[:, :, 1]*mask_temp_np, (scale*image_resize, scale*image_resize),interpolation=cv2.INTER_LINEAR)
            gt_masked_flow = flow_viz_np(gt_flow_masked_x, gt_flow_masked_y)
            if args.render:
                cv2.imshow('GT Masked Flow', cv2.cvtColor(gt_masked_flow, cv2.COLOR_BGR2RGB))
             
            cv2.waitKey(1)

       

        AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(gt_flow, pred_flow, (torch.sum(torch.sum(torch.sum(event_data, dim=0), dim=0), dim=0)).cpu(), is_car=False)

        AEE_sum = AEE_sum + args.div_flow * AEE
        AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

        AEE_sum_gt = AEE_sum_gt + args.div_flow * AEE_gt
        AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

        percent_AEE_sum += percent_AEE

         # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        iters += 1

    print('-------------------------------------------------------')
    print('Mean AEE: {:.2f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, mean %AEE: {:.3f}, # pts: {:.2f}'
                  .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_AEE_sum / iters, n_points))
    print('-------------------------------------------------------')
    

    return AEE_sum / iters



def main():
    global args, best_EPE, image_resize, device
    save_path = '{},{},{}epochs{},b{},lr{},loss{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
        args.loss)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.savedir,save_path)
    print('=> Everything will be saved to {}'.format(save_path))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    event_dir = testdir + '/event_data'
    imgs_dir = testdir  + '/gray_data'
    dir_gt = testdir  + '/gt_data'
    vox_transforms_list = [CenterCrop(size=256)]
    Test_dataset = TrainDataset(event_dir, imgs_dir, dir_gt, data_split, image_resize,vox_transforms_list,100)
    test_loader = DataLoader(dataset=Test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        #args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))
        
    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay},
                    {'params': log_var_a},
                    {'params': log_var_b}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return

if __name__ == '__main__':
    main()
