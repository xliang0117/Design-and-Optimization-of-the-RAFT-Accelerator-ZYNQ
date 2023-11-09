from operator import mod
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import brevitas
from brevitas.export import export_brevitas_onnx


DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float() # get the tensor 
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    # flo = flow_viz.flow_to_image(flo)
    norm_flo = flo / (np.abs(flo).max())
    w,h,_ = norm_flo.shape
    rgb_map = np.ones((w,h,3)).astype(np.float32)
    rgb_map[:,:,0] += norm_flo[:,:,0]
    rgb_map[:,:,1] -= 0.5*(norm_flo[:,:,0] + norm_flo[:,:,1])
    rgb_map[:,:,2] += norm_flo[:,:,1]
    rgb_map = rgb_map.clip(0,1) * 255
    
    img_flo = np.concatenate([img, rgb_map], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):

    brevitas.config.IGNORE_MISSING_KEYS = True    

    model = RAFT(args)       # use multiple gpu if exists 
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))   # load the module parameter

    # model = model.module    # get the real module after dataparallel()
    model.to(DEVICE)        # set the target device //CPU or GPU
    model.eval()            # set the module in evaluation mode


    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            images = torch.cat((image1, image2), dim=0)
            # export_brevitas_onnx(model, input_t=images, export_path="onnx_model.onnx")

            flow_low, flow_up, _ = model(images, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--del_norm', action='store_true', help='del norm')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
