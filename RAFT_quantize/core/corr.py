import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler
from extractor import Activation


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.actFirst = Activation(True, name="corr0", logType=3, input_scale=0.015625*0.015625, weight_scale=1.0, output_scale=0.03125)
        self.act = torch.nn.ModuleList([Activation(True, name="corr{}".format(x), logType=3, input_scale=1.0, weight_scale=1.0, output_scale=1.0) 
                                        for x in range(1,4)])

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_truncate = self.actFirst(torch.reshape(corr, (1, h1*w1, h2, w2)))
        corr_truncate = corr_truncate.reshape(h1*w1, 1, h2, w2)
        self.corr_pyramid.append(corr_truncate)

        for i in range(self.num_levels-1):
            corr_truncate = F.avg_pool2d(corr_truncate, 2, stride=2)
            corr_truncate = self.act[i](torch.reshape(corr_truncate, ( 1, h1*w1, h2//(2**(i+1)), w2//(2**(i+1)) )))
            corr_truncate = corr_truncate.reshape(h1*w1, 1, h2//(2**(i+1)), w2//(2**(i+1)))
            self.corr_pyramid.append(corr_truncate)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
