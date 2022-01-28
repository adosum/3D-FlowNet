import torch.nn as nn
import torch.nn.functional as F
import torch
# try:
#     from spatial_correlation_sampler import spatial_correlation_sample
# except ImportError as e:
#     import warnings
#     with warnings.catch_warnings():
#         warnings.filterwarnings("default", category=ImportWarning)
#         warnings.warn("failed to load custom correlation module"
#                       "which is needed for FlowNetC", ImportWarning)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        )


def predict_flow(batchNorm, in_planes):
    if batchNorm:
        return nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_planes,2,kernel_size=1,stride=1,padding=0,bias=True),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Tanh()
        )


def deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
    
def conv_3d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,padding=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    

def deconv_3d(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(in_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))
    