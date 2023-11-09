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
        self.fnet = SmallEncoder(output_dim=128, norm_fn='batch')        
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

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


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

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
