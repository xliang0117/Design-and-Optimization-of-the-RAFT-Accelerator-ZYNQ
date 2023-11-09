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

import sys


class ConstFlowQuant(
    NarrowIntQuant, PerTensorConstScaling2bit, ActQuantSolver):
    bit_width = 8
    # scaling_const = 0.5 # 2**-5
    tensor_clamp_impl = TensorClamp
    min_val = -3.96875
    max_val = 3.96875

class ConstFnetConv2Quant(
    NarrowIntQuant, PerTensorConstScaling2bit, ActQuantSolver):
    bit_width = 8
    # scaling_const = 0.5 # 2**-6
    tensor_clamp_impl = TensorClamp
    min_val = -1.984375
    max_val = 1.984375

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.depthconv1 = qnn.QuantConv2d(input_dim, input_dim, 3, padding=1, groups=input_dim, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.pointconv1 = qnn.QuantConv2d(input_dim, hidden_dim, 1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(hidden_dim, 2, 1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, output_quant=ConstFlowQuant, return_quant_tensor=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.pointconv1(self.depthconv1(x))))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.depthconvz = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.pointconvz = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.depthconvr = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.pointconvr = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.depthconvq = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.pointconvq = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.sigmoid1 = qnn.QuantSigmoid(act_quant=ActQuant, return_quant_tensor=False)
        self.sigmoid2 = qnn.QuantSigmoid(act_quant=self.sigmoid1.act_quant, return_quant_tensor=True)
        self.tanh = qnn.QuantTanh(input_quant=ConstFnetConv2Quant,act_quant=ConstFlowQuant, return_quant_tensor=True)
        self.quantize = qnn.QuantIdentity(act_quant=ConstFlowQuant, return_quant_tensor=True)
        self.elementAdd = qnn.QuantEltwiseAdd(input_quant=ConstFlowQuant, output_quant=ConstFlowQuant, return_quant_tensor=True)
        self.cat = qnn.QuantCat(input_quant=ConstFlowQuant, output_quant=ConstFlowQuant, return_quant_tensor=True)


    def forward(self, h, x):
        hx = self.cat([h, x], dim=1)

        z = self.sigmoid1(self.pointconvz(self.depthconvz(hx)))
        r = self.sigmoid2(self.pointconvr(self.depthconvr(hx)))
        rh = self.quantize(r * h)
        q = self.tanh(self.pointconvq(self.depthconvq(self.cat([rh, x], dim=1))))
        
        minus_z = 1-z
        hz = minus_z * h
        zq = z * q

        h = self.elementAdd(hz, zq)
        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = qnn.QuantConv2d(cor_planes, 96, 1, padding=0, weight_quant=WeightQuant, bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=True)
        self.convf1 = qnn.QuantConv2d(2, 64, 1, padding=0, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.convf2 = qnn.QuantConv2d(64, 32, 1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, output_quant=self.convc1.output_quant, return_quant_tensor=True)
        self.depthconv = qnn.QuantConv2d(128, 128, 3, padding=1, groups=128, input_quant=self.convc1.output_quant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.pointconv = qnn.QuantConv2d(128, 80, 1, padding=0, input_quant=ActQuant, weight_quant=WeightQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.cat = qnn.QuantCat(input_quant=ConstFlowQuant, output_quant=ConstFlowQuant, return_quant_tensor=True)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = self.cat([cor, flo], dim=1)
        out = F.relu(self.pointconv(self.depthconv(cor_flo)))
        return self.cat([out, flow], dim=1)
    


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.quantize = qnn.QuantIdentity(act_quant=ConstFlowQuant, return_quant_tensor=True)
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=96)
        self.cat = qnn.QuantCat(input_quant=ConstFlowQuant, output_quant=ConstFlowQuant, return_quant_tensor=True)

    def forward(self, net, inp, corr, flow):
        net = self.quantize(net)
        inp = self.quantize(inp)
        corr = self.quantize(corr)
        flow = self.quantize(flow)        

        motion_features = self.encoder(flow, corr)
        inp = self.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
                
        return net, None, delta_flow


