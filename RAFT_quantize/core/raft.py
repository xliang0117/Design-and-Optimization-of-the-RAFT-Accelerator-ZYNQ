from tkinter import image_names
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import SmallUpdateBlock
from extractor import SmallEncoder
from corr import CorrBlock
from utils.utils import coords_grid, upflow8
import time

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        args.corr_levels = 4
        args.corr_radius = 3
    

        # feature network, context network, and update block
        fnet_norm = 'none' if args.del_norm else 'batch'
        self.fnet = SmallEncoder(output_dim=64, norm_fn=fnet_norm)        
        self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none')
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1


    def forward(self, images, iters=1, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        batch_dim = images.shape[0] // 2
        images = torch.split(images, [batch_dim, batch_dim], dim=0)

        image1 = 2 * (images[0] / 255.0) - 1.0
        image2 = 2 * (images[1] / 255.0) - 1.0

        image1 = image1.contiguous()    # make the tensor contiguous in memory
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim


        # run the feature network
        with autocast(enabled=self.args.mixed_precision):   # mixed precision usage, like float16 mixed float32, for training
            fmap1, fmap2 = self.fnet([image1, image2])      # extract the feature

        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)    # extract the context
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)


        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        # coords1 = coords1 + (2**1e-8)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # get frequency of flow
            if test_mode:
                delta_coords = torch.floor(coords1 + delta_flow) - torch.floor(coords1) # if itr != 0 else torch.zeros(coords1.shape)
                delta_coords = torch.sum(torch.abs(delta_coords), dim=1)
                histOutOriginalflow, rangeBin = np.histogram(delta_coords.cpu(), bins=(0,1,2,3,4,5,6,7,8,9,10,100))
                # print("origin:{}".format(histOutOriginalflow))

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = upflow8(coords1 - coords0)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, histOutOriginalflow
            
        return flow_predictions
