import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int16Bias as BiasQuant
from brevitas.quant.base import NarrowIntQuant, PerTensorConstScaling2bit
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.core.function_wrapper import TensorClamp
from brevitas.nn.utils import merge_bn

class ConstInputQuant(
    NarrowIntQuant, PerTensorConstScaling2bit, ActQuantSolver):
    bit_width = 8
    tensor_clamp_impl = TensorClamp
    min_val = -0.9921875    # 2**-7
    max_val = 0.9921875


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='batch', stride=1):
        super(BottleneckBlock, self).__init__()
        self.norm_fn = norm_fn
          
        self.conv1 = qnn.QuantConv2d(in_planes, 32, kernel_size=1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.depthconv2 = qnn.QuantConv2d(32, 32, kernel_size=3, padding=1, stride=stride, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, groups=32, return_quant_tensor=True)
        self.pointconv2 = qnn.QuantConv2d(32, 32, kernel_size=1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(32, planes, kernel_size=1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.conv4 = qnn.QuantConv2d(in_planes, planes, kernel_size=1, stride=stride, input_quant=self.conv1.input_quant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.elementAdd = qnn.QuantEltwiseAdd(input_quant=ActQuant, output_quant=None, tie_input_output_quant=True, return_quant_tensor=True)
        
        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
            self.norm2 = nn.BatchNorm2d(32)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = False
        else:    
            self.downsample = True


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.pointconv2(self.depthconv2(y))))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample:
            x = self.norm4(self.conv4(x))

        return self.relu(self.elementAdd(x, y))


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch'):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        
        self.quant_inp = qnn.QuantIdentity(act_quant=ActQuant, return_quant_tensor=True)

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.depthconv1 = qnn.QuantConv2d(3, 3, kernel_size=3, stride=2, padding=1, input_quant=ConstInputQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, groups=3, return_quant_tensor=True)
        self.pointconv1 = qnn.QuantConv2d(3, 32, kernel_size=1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        # self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        
        self.conv2 = qnn.QuantConv2d(96, output_dim, kernel_size=1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=False)


    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        # layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(layer1)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            
        x = self.pointconv1(self.depthconv1(x))
        x = self.norm1(x)
        x = self.relu1(x)

        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x