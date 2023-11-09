import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.depthconv1 = qnn.QuantConv2d(input_dim, input_dim, 3, padding=1, groups=input_dim, weight_bit_width=8)
        self.pointconv1 = qnn.QuantConv2d(input_dim, hidden_dim, 1, weight_bit_width=8)
        self.depthconv2 = qnn.QuantConv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, weight_bit_width=8)
        self.pointconv2 = qnn.QuantConv2d(hidden_dim, 2, 1, weight_bit_width=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.pointconv2(self.depthconv2(self.relu(self.pointconv1(self.depthconv1(x)))))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.depthconvz = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_bit_width=8)
        self.pointconvz = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, weight_bit_width=8)
        self.depthconvr = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_bit_width=8)
        self.pointconvr = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, weight_bit_width=8)
        self.depthconvq = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1, groups=hidden_dim+input_dim, weight_bit_width=8)
        self.pointconvq = qnn.QuantConv2d(hidden_dim+input_dim, hidden_dim, 1, weight_bit_width=8)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.pointconvz(self.depthconvz(hx)))
        r = torch.sigmoid(self.pointconvr(self.depthconvr(hx)))
        q = torch.tanh(self.pointconvq(self.depthconvq(torch.cat([r*h, x], dim=1))))

        h = (1-z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = qnn.QuantConv2d(cor_planes, 96, 1, padding=0, weight_bit_width=8)
        self.convf1 = qnn.QuantConv2d(2, 64, 1, weight_bit_width=8)
        self.convf2 = qnn.QuantConv2d(64, 32, 1, weight_bit_width=8)
        self.depthconv = qnn.QuantConv2d(128, 128, 3, padding=1, groups=128, weight_bit_width=8, bias=False)
        self.pointconv = qnn.QuantConv2d(128, 80, 1, weight_bit_width=8)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.pointconv(self.depthconv(cor_flo)))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


