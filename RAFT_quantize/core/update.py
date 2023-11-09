import torch
import torch.nn as nn
import torch.nn.functional as F
from extractor import Activation
import numpy as np
import math
import sys

class TensorLookUp(nn.Module):
    def __init__(self, act, name, logType, bitwidth=8, func=0, quick=True, truncate=True, test=False) -> None:
        super(TensorLookUp, self).__init__()
        self.lookUpTable = {}
        self.act = act
        self.name = name
        self.logType = logType
        self.bitwidth=bitwidth
        self.func = func
        self.quick = quick
        self.truncate = truncate
        self.test = test

        sig_input_scale = 0.015625
        sig_output_scale = 0.015625
        tanh_input_scale = 0.015625
        tanh_output_scale = 0.03125
        self.input_scale = torch.tensor(sig_input_scale) if func == 0 else torch.tensor(tanh_input_scale)
        self.output_scale = torch.tensor(sig_output_scale) if func == 0 else torch.tensor(tanh_output_scale)
        self.calFunc = torch.sigmoid if func == 0 else torch.tanh

        for addr in range(-2**(bitwidth-1), 2**(bitwidth-1)):
            actualValue = torch.tensor(addr) * self.input_scale
            funcValue = self.calFunc(actualValue) / self.output_scale
            self.lookUpTable[addr] = torch.clamp(torch.floor(funcValue + 0.5), min=-2**(bitwidth-1), max=2**(bitwidth-1)-1)

    def forward(self, x):
        if self.test and self.act:
            y = self.calFunc(x*self.input_scale)/self.output_scale
            y = torch.clamp(torch.floor(y + 0.5), min=-2**(self.bitwidth-1), max=2**(self.bitwidth-1)-1)
            x.apply_(self.lookUpTable.get)
            print("sig/tanh test error:{}".format(torch.sum(y-x)))

        elif self.act:
            if self.logType == 1:
                torch.save(self.calFunc(x*self.input_scale), "export/log/"+self.name+".pt")
            if self.quick:
                x = self.calFunc(x*self.input_scale)/self.output_scale
                x = torch.clamp(torch.floor(x + 0.5), min=-2**(self.bitwidth-1), max=2**(self.bitwidth-1)-1) if self.truncate else x
            else:
                x.apply_(self.lookUpTable.get)
            if self.logType == 3:
                Activation.ch_div32_save(x, self.name)

        else:
            x = self.calFunc(x)
            if self.logType == 2:
                torch.save(x, "export/log/"+self.name+"_compare.pt")
                data4Compare = torch.load("export/log/"+self.name+".pt", map_location=torch.device('cpu'))
                xMean = torch.mean(torch.abs(x)) if torch.mean(torch.abs(x)) != 0 else torch.tensor(1.0)
                error = torch.mean(torch.abs((x - data4Compare)))  / xMean * 100
                print("{:<15} error:{:.3f}%".format(self.name, error))
        return x    


class FlowHead(nn.Module):
    def __init__(self, args, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.depthconv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim)
        self.d1act = Activation(args.act, "fh-d1", args.logType)
        self.pointconv1 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)
        self.p1act = Activation(args.act, "fh-p1", args.logType)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 1, padding=0)
        self.c2act = Activation(args.act, "fh-c2", args.logType, outQuant=True)
        # self.c2logact = Activation(args.act, "fh-flow", args.logType, outQuant=True, input_scale=0.03125, weight_scale=0.0078125, output_scale=0.03125)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, flow):
        x = self.d1act(self.depthconv1(x))
        x = self.p1act(self.relu(self.pointconv1(x)))
        x = self.c2act(self.conv2(x))
        Activation.ch_div32_save(flow+x, "final-flow")
        return x*0.03125

class ConvGRU(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.act = args.act
        self.depthconvz = nn.Conv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim)
        self.dzact = Activation(args.act, "gru-dz", args.logType)
        self.pointconvz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 1)
        self.pzact = Activation(args.act, "gru-pz", args.logType)
        self.depthconvr = nn.Conv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim)
        self.dract = Activation(args.act, "gru-dr", args.logType)
        self.pointconvr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 1)
        self.pract = Activation(args.act, "gru-pr", args.logType)
        self.depthconvq = nn.Conv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim)
        self.dqact = Activation(args.act, "gru-dq", args.logType)
        self.pointconvq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 1)
        self.pqact = Activation(args.act, "gru-pq", args.logType)
        self.sig1 = TensorLookUp(args.act, "gru-sig1", args.logType, func=0)
        self.sig2 = TensorLookUp(args.act, "gru-sig2", args.logType, func=0)
        self.tanh = TensorLookUp(args.act, "gru-tanh", args.logType, func=1)
        self.quantize = Activation(args.act, "gru-constquant", args.logType, input_scale=self.sig1.output_scale*0.03125, weight_scale=1, output_scale=0.03125)
        self.qhact = Activation(args.act, "gru-qh", args.logType, input_scale=0.03125, weight_scale=1.0, output_scale=0.03125)
        self.zqact = Activation(args.act, "gru-zq", args.logType, input_scale=self.sig1.output_scale*0.03125, weight_scale=1, output_scale=0.03125)
        self.netact = Activation(args.act, "gru-outnet", args.logType, input_scale=0.03125, weight_scale=1, output_scale=0.03125)


    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = self.sig1(self.pzact(self.pointconvz(self.dzact(self.depthconvz(hx)))))
        r = self.sig2(self.pract(self.pointconvr(self.dract(self.depthconvr(hx)))))
        rh = self.quantize(r * h)
        q = self.tanh(self.pqact(self.pointconvq(self.dqact(self.depthconvq(torch.cat([rh, x], dim=1))))))
        
        q_minus_net = self.qhact(q - h)
        z_mul_qnet = self.zqact(z * q_minus_net) 
        h = self.netact(z_mul_qnet + h)

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.cc1act = Activation(args.act, "me-cc1", args.logType)
        self.convf1 = nn.Conv2d(2, 64, 1, padding=0)
        self.cf1act = Activation(args.act, "me-cf1", args.logType)
        self.convf2 = nn.Conv2d(64, 32, 1, padding=0)
        self.cf2act = Activation(args.act, "me-cf2", args.logType)
        self.depthconv = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.dnvact = Activation(args.act, "me-dnv", args.logType)
        self.pointconv = nn.Conv2d(128, 80, 1, padding=0)
        self.pnvact = Activation(args.act, "me-pnv", args.logType)

    def forward(self, flow, corr):
        cor = self.cc1act(F.relu(self.convc1(corr)))
        flo = self.cf1act(F.relu(self.convf1(flow)))
        flo = self.cf2act(F.relu(self.convf2(flo)))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.pnvact(F.relu(self.pointconv(self.dnvact(self.depthconv(cor_flo)))))
        return torch.cat([out, flow], dim=1)
    
def save_corr(corr):
    corr = corr[0,:,:,:].numpy().astype(np.int8)
    corr_part = np.split(corr, 4, axis=0)
    corr = np.concatenate([np.transpose(x.reshape(7,7,48,48), (1,0,2,3)).reshape(49, 48, 48) for x in corr_part], axis=0)
    corr_part = np.split(corr, 4, axis=0)
    new_slice = []
    new_slice.append(np.concatenate([x[0:32,:,:] for x in corr_part], axis=0))
    new_slice.append(np.concatenate([x[32:48,:,:] for x in corr_part], axis=0))
    new_slice.append(np.stack([x[48,:,:] for x in corr_part], axis=0))
    new_corr = np.concatenate(new_slice, axis=0)
    new_corr = np.pad(new_corr, ((0, 28), (0, 0), (0, 0)), 'constant')
    new_corr = np.transpose(new_corr, (1,2,0))
    with open("export/"+"new-corr"+".bin", "wb") as f:
        x_split = np.split(new_corr, 7, axis=2)
        for part in x_split:
            f.write(part.tobytes())

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        fnetConv2Scale = 0.015625
        self.netQ = Activation(args.act, "u-net", args.logType, input_scale=0.03125, weight_scale=1.0, output_scale=0.03125)
        self.inpQ = Activation(args.act, "u-inp", args.logType, input_scale=fnetConv2Scale, weight_scale=1.0, output_scale=0.03125)
        self.corrQ = Activation(args.act, "u-corr", args.logType, input_scale=1.0, weight_scale=1.0, output_scale=1.0)
        self.flowQ = Activation(args.act, "u-flow", args.logType, input_scale=1.0, weight_scale=1.0, output_scale=0.03125)
        self.dequantize = Activation(args.act, "deConstQuant", args.logType, input_scale=0.03125, weight_scale=1, output_scale=0.03125, truncate=False)
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(args, hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(args, hidden_dim, hidden_dim=96)

    def forward(self, net, inp, corr, flow):
        net = self.netQ(net)
        inp = self.inpQ(inp)
        corr = self.corrQ(corr+2**-11)
        flow = self.flowQ(flow)       
        save_corr(corr) 

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net, flow)
        
        net = self.dequantize(net)
        return net, None, delta_flow

