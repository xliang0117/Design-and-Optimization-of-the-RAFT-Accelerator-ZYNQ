import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int16Bias as BiasQuant
from brevitas.nn.utils import merge_bn

class Activation(nn.Module):
    def __init__(self, act, name, logType, input_scale=0.0, weight_scale=0.0, output_scale=0.0, inBitwidth=16, outBitwidth=8, 
                signed=True, outQuant=True, truncate=True):
        super(Activation, self).__init__()
        # this scale should be s_inp*s_weight/s_out
        self.act = act
        self.logType = logType
        self.name = name
        self.input_scale = torch.tensor(input_scale)
        self.weight_scale = torch.tensor(weight_scale)
        self.output_scale = torch.tensor(output_scale)
        if self.act:
            if input_scale == 0.0 and weight_scale == 0.0 and output_scale == 0.0:
                self.input_scale = nn.Parameter(torch.tensor(input_scale))
                self.weight_scale = nn.Parameter(torch.tensor(weight_scale))
                self.output_scale = nn.Parameter(torch.tensor(output_scale))
            self.inBitwidth = inBitwidth
            self.inLimit = (2**(self.inBitwidth-1)-1,-2**(self.inBitwidth-1)) if signed else (2**(self.inBitwidth)-1, 0)
            self.outBitwidth = outBitwidth
            self.outLimit = (-2**(self.outBitwidth-1), 2**(self.outBitwidth-1)-1) if signed else (0, 2**(self.outBitwidth)-1)
            self.signed = signed
            self.truncate = truncate
            self.outQuant = outQuant

    @staticmethod
    def ch_div32_save(x, name):
        x_np = x[0,:,:,:].numpy().astype(np.int8)
        pad_ch = (x_np.shape[0] + 32 - 1) // 32 * 32
        x_np = np.pad(x_np, ((0, pad_ch-x_np.shape[0]), (0, 0), (0, 0)), 'constant')
        x_np = np.transpose(x_np, (1,2,0))
        with open("export/"+name+".bin", "wb") as f:
            x_split = np.split(x_np, pad_ch//32, axis=2)
            for part in x_split:
                f.write(part.tobytes())

    def forward(self, x):
        # this scale not change the value back to real, but to next layer's quantized input
        # so after this scale, the x should still be integer
        if self.act:
            x = x * self.input_scale * self.weight_scale
            if self.logType == 1:
                torch.save(x, "export/log/"+self.name+".pt")
            if self.outQuant:
                x = x / self.output_scale
                x = torch.floor(x) if self.truncate else x # make those not integer number turn to zero
                x = torch.clamp(x, min=self.outLimit[0], max=self.outLimit[1]) if self.truncate else x
            if self.logType == 3:
                self.ch_div32_save(x, self.name)
        if self.logType == 2:
            torch.save(x, "export/log/"+self.name+"_compare.pt")
            data4Compare = torch.load("export/log/"+self.name+".pt", map_location=torch.device('cpu'))
            xMean = torch.mean(torch.abs(x)) if torch.mean(torch.abs(x)) != 0 else torch.tensor(1.0)
            error = torch.mean(torch.abs((x - data4Compare)))  / xMean * 100
            print("{:<15} error:{:.3f}%".format(self.name, error))

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, name="", act=True, logType=0):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.depthconv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride, groups=planes//4)
        self.pointconv2 = nn.Conv2d(planes//4, planes//4, kernel_size=1)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        self.c1act = Activation(act, name+"."+"c1act", logType)
        self.d2act = Activation(act, name+"."+"d2act", logType)
        self.p2act = Activation(act, name+"."+"p2act", logType)
        self.c3act = Activation(act, name+"."+"c3act", logType)
        self.c4act = Activation(act, name+"."+"c4act", logType)
        
        if stride == 1:
            self.downsample = False
        else:    
            self.downsample = True


    def forward(self, x):
        y = x
        y = self.c1act(self.relu(self.conv1(y)))
        y = self.p2act(self.relu(self.pointconv2(self.d2act(self.depthconv2(y)))))
        y = self.c3act(self.relu(self.conv3(y)))
        if self.downsample:
            x = self.c4act(self.conv4(x))

        if self.c4act.logType == 3:
            Activation.ch_div32_save(torch.clamp(self.relu(x+y), min=-128, max=127), self.c4act.name)

        return torch.clamp(self.relu(x+y), min=-128, max=127)


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, act=True, logType=0):
        super(SmallEncoder, self).__init__()
        self.act = act
        self.logType = logType
        
        self.depthconv1 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3)
        self.d1act = Activation(act, "d1act", logType)
        self.pointconv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.p1act = Activation(act, "p1act", logType)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        # self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2, name="layer2")
        self.layer3 = self._make_layer(96, stride=2, name="layer3")
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        self.c2act = Activation(act, "c2act", logType)


    def _make_layer(self, dim, name="", stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, name=name, stride=stride, act=self.act, logType=self.logType)
        # layer2 = BottleneckBlock(dim, dim, stride=1)
        # layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(layer1)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.d1act(self.depthconv1(x))
        x = self.p1act(self.pointconv1(x))
        x = self.relu1(x)

        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.c2act(self.conv2(x))

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x