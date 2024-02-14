#!/usr/bin/python3

import argparse
import sys
import os
import time

import torchvision.transforms as transforms

import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from models import Generator_F2S

torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/home/yangyang/code/shadow_removal/demo3/ckptluoma/netG_S2F_48.pth',
                    help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/home/yangyang/code/shadow_removal/demo3/ckptluoma/netG_refine_48.pth',
                    help='B2A generator checkpoint file')
opt = parser.parse_args()

### ISTD
opt.dataroot_A = '/home/yangyang/code/shadow_removal/demo3/AISD2/test/test/test/test_shadow'  # 只有阴影区域
opt.dataroot_B = '/home/yangyang/code/shadow_removal/demo3/AISD2/test/test/test/test_mask'  # 残差区域
opt.dataroot_C = '/home/yangyang/code/shadow_removal/demo3/AISD2/test/test/test/dil'  # 残差区域
opt.dataroot_D = '/home/yangyang/code/shadow_removal/demo3/AISD2/test/test/test/dil' #残差区域
opt.im_suf_A = '.png'
opt.im_suf_B = '.png'
opt.im_suf_C = '.png'
opt.im_suf_D = '.png'

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda')

print(opt)

img_transform1 = transforms.Compose([
    #    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
img_transform3 = transforms.Compose([
    transforms.ToTensor(),
])
###### Definition of variables ######
# Networks
netG_S2F = Generator_F2S()
# style=StyleEncoder(16,norm_layer=nn.InstanceNorm2d)
netG_refine = Generator_F2S()
if opt.cuda:
    netG_S2F = netG_S2F.cuda()
    netG_refine = netG_refine.cuda()

    netG_S2Fcheckpoint = torch.load(opt.generator_A2B, map_location='cuda')
    netG_S2F.load_state_dict(netG_S2Fcheckpoint)
    #netG_S2F.load_state_dict(torch.load(opt.generator_A2B , map_location='cuda'))
    # style.load_state_dict(torch.load(opt.style,map_location='cuda'))
    netG_refine.load_state_dict(torch.load(opt.generator_B2A, map_location='cuda'))

    # Set model's test mode
    netG_S2F.eval()

    # style.eval()
    netG_refine.eval()

    # Dataset loader

    to_pil = transforms.ToPILImage()

    ###### Testing######

    # Create output dirs if they don't exist
    # if not os.path.exists('output/A'):
    #    os.makedirs('output/A')
    if not os.path.exists('ckpt/Bb' ):
        os.makedirs('ckpt/Bb' )

    ##################################### A to B // shadow to shadow-free
    gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

    # mask_queue = QueueMask(gt_list.__len__())

    # mask_non_shadow = Variable(Tensor(1, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False)
    start_time=time.time()
    for idx, img_name in enumerate(gt_list):
        if idx==6:
            end_time=time.time()
            print(end_time-start_time)
        print('predicting: %d / %d' % (idx + 1, len(gt_list)))
        print(img_name)
        # Set model input
        # img_orignal=Image.open(os.path.join(opt.dataroot_D, img_name + opt.im_suf_A)).convert('RGB') #原图
        img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')  # 只有阴影区域的图像
        w, h = img.size
        img = (img_transform1(img).unsqueeze(0)).to(device)
        # img_res = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_A)).convert('RGB')
        mask = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))

        maskn = (img_transform2(mask).unsqueeze(0)).to(device)  ##用来合成最后的图像的掩模图

        mask1 = maskn[:, 0, :, :].unsqueeze(0)  # -1-1的阴影掩模图

        mask2 = (img_transform3(mask).unsqueeze(0)).to(device)
        mask22 = mask2[:, 0, :, :].unsqueeze(0)  # 0-1的阴影掩膜图

        mask_dil = Image.open(os.path.join(opt.dataroot_D, img_name + opt.im_suf_D))
        mask_dil = (img_transform2(mask_dil).unsqueeze(0)).to(device)
        mask_dill = mask_dil[:, 0, :, :].unsqueeze(0)


        #temp_B = netG_S2F(img, mask1)  # temp_B是只有阴影区域的

        #temp_B = temp_B * mask2 + img * ((mask2 - 1) * (-1))
        all = netG_refine(img, mask_dill)

        # 下面根据掩膜的图像来判断选择原始图像还是生成的阴影图吧

        imgfinal = img * ((mask2 - 1) * (-1)) + all * mask2

        imgfinal = 0.5 * (imgfinal.data + 1.0)
        # mask_queue.insert(mask_generator(img_var, temp_B))
        imgfinal = np.array((to_pil(imgfinal.squeeze(0).cpu())))
        Image.fromarray(imgfinal).save('ckpt/Bb/'+  img_name + opt.im_suf_A)

        print('Generated images %04d of %04d' % (idx + 1, len(gt_list)))
