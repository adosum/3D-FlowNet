import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn



"""
Calculates per pixel flow error between flow_pred and flow_gt. event_img is used to mask out any pixels without events
"""
def flow_error_dense(flow_gt, flow_pred, event_img, is_car=False):
    max_row = flow_gt.shape[1]
    if is_car == True:
        max_row = 190

    flow_pred = np.array(flow_pred)
    event_img = np.array(event_img)
    assert event_img.shape[0] == event_img.shape[1], \
            'The event mask has wrong size, flow error wrong' 
    event_img_cropped = np.squeeze(event_img)[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :]
    flow_pred_cropped = flow_pred[:max_row, :]

    event_mask = event_img_cropped > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])), np.linalg.norm(flow_gt_cropped, axis=2) > 0)
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))

    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    EE_gt = np.linalg.norm(gt_masked, axis=-1)

    n_points = EE.shape[0]

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    EE = torch.from_numpy(EE)
    EE_gt = torch.from_numpy(EE_gt)

    if torch.sum(EE) == 0:
        AEE = 0
        AEE_sum_temp = 0

        AEE_gt = 0
        AEE_sum_temp_gt = 0
    else:
        AEE = torch.mean(EE)
        AEE_sum_temp = torch.sum(EE)

        AEE_gt = torch.mean(EE_gt)
        AEE_sum_temp_gt = torch.sum(EE_gt)

    return AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt


"""Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow. x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement."""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    return


"""The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity. Pseudo code for this process is as follows:
x_orig = range(cols)      y_orig = range(rows)
x_prop = x_orig           y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)
The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.
Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at each timestamp.
  gt_timestamps - timestamp for each flow array.  start_time, end_time - gt flow will be estimated between start_time and end time."""
def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_timestamps, start_time, end_time):

    x_flow_in = np.array(x_flow_in, dtype=np.float64)
    y_flow_in = np.array(y_flow_in, dtype=np.float64)

    gt_timestamps = np.array(gt_timestamps, dtype=np.float64)
    start_time = np.array(start_time, dtype=np.float64)
    end_time = np.array(end_time, dtype=np.float64)

    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between gt_iter and gt_iter+1.
    gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt > dt:
        return x_flow*dt/gt_dt, y_flow*dt/gt_dt

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)
    gt_iter += 1

    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return x_shift, y_shift



def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        # weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        weights = [0.01, 0.02, 0.08, 0.32]
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)
