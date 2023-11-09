import sys
sys.path.append("./script")

from code_gen import hlsGenerator
from file_gen import fileWrapper, fileTemplate
import argparse

import colored_traceback
colored_traceback.add_hook()


ddrConfig = [
    # static
    {"name":"PRESERVE",     "ch":1,   "height":1,     "width": 385,   "start":None},

    # dynamic
    {"name":"E-A-3",        "ch":3,   "height":192,   "width": 192,   "start":None},
    {"name":"E-A-1",        "ch":32,  "height":192,   "width": 192,   "start":None},
    {"name":"E-E",          "ch":16,  "height":192,   "width": 192,   "start":None},
    {"name":"E-F-3",        "ch":16,  "height":192,   "width": 192,   "start":None},
    {"name":"E-F-1",        "ch":16,  "height":96,    "width": 96,    "start":"E-E"},
    {"name":"E-G",          "ch":64,  "height":96,    "width": 96,    "start":None},
    {"name":"E-EP1",        "ch":64,  "height":96,    "width": 96,    "start":None},
    {"name":"E-H",          "ch":24,  "height":96,    "width": 96,    "start":"E-A-3"},
    {"name":"E-I-3",        "ch":24,  "height":96,    "width": 96,    "start":None},
    {"name":"E-I-1",        "ch":24,  "height":48,    "width": 48,    "start":"E-H"},
    {"name":"E-J",          "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"E-EP2",        "ch":96,  "height":48,    "width": 48,    "start":None},

    {"name":"corr",         "ch":196, "height":48,    "width": 48,    "start":"E-A-3"},
    {"name":"cor",          "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"flo2",         "ch":32,  "height":48,    "width": 48,    "start":None},
    {"name":"flo1",         "ch":64,  "height":48,    "width": 48,    "start":None},
    {"name":"net",          "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"inp",          "ch":64,  "height":48,    "width": 48,    "start":None},
    {"name":"out",          "ch":80,  "height":48,    "width": 48,    "start":None},
    # {"name":"flow",         "ch":2,   "height":48,    "width": 48,    "start":None},
    {"name":"q_pre",        "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"z",            "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"f_head",       "ch":96,  "height":48,    "width": 48,    "start":None},

    {"name":"E-fmap1",      "ch":96,  "height":48,    "width": 64,    "start":"static"},
    {"name":"E-fmap2",      "ch":96,  "height":48,    "width": 64,    "start": None},
    {"name":"Corr1",        "ch":2304,"height":48,    "width": 48,    "start": None},
    {"name":"Corr2",        "ch":2304,"height":24,    "width": 24,    "start": None},
    {"name":"Corr3",        "ch":2304,"height":12,    "width": 12,    "start": None},
    {"name":"Corr4",        "ch":2304,"height":6,     "width": 6,     "start": None},
]

layerConfig = [
{ "split":1},
{ "name":"EA3"  , "inc": 3  , "outc":3  , "sde":2,  "dw":1, "pw":0, "sl":[9,5], "inp":"IMAGE1", "out":"E-A-3",  "relu":0 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"EA1"  , "inc": 3  , "outc":32 , "sde":1,  "dw":0, "pw":1, "sl":[9,5], "inp":"E-A-3",  "out":"E-A-1",  "relu":1 ,"in_dim":192 }, # fnet.pointconv1
{ "name":"EE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A-1",  "out":"E-E",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"EF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":0, "sl":[5,7], "inp":"E-E",    "out":"E-F-3",  "relu":0 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"EF1"  , "inc": 16 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[5,7], "inp":"E-F-3",  "out":"E-F-1",  "relu":1 ,"in_dim":96  }, # fnet.layer2.0.pointconv2
{ "name":"EG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-F-1",  "out":"E-G",    "relu":1 ,"in_dim":96  }, # fnet.layer2.0.conv3
{ "name":"EP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A-1",  "out":"E-EP1",  "relu":0 ,"in_dim":192  , "eop":[["load", "afm_2", "E-G"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer2.0.conv4
{ "name":"EH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP1",  "out":"E-H",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"EI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":0, "sl":[6,7], "inp":"E-H",    "out":"E-I-3",  "relu":0 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"EI1"  , "inc": 24 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[6,7], "inp":"E-I-3",  "out":"E-I-1",  "relu":1 ,"in_dim":48  }, # fnet.layer3.0.pointconv2
{ "name":"EJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-I-1",  "out":"E-J",    "relu":1 ,"in_dim":48  }, # fnet.layer3.0.conv3
{ "name":"EP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP1",  "out":"E-EP2",  "relu":0 ,"in_dim":96   , "eop":[["load", "afm_2", "E-J"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer3.0.conv4
{ "name":"EK"   , "inc": 96 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP2",  "out":"E-fmap1","relu":0 ,"in_dim":48  }, # fnet.conv2


{ "name":"CA3"  , "inc": 3  , "outc":3  , "sde":2,  "dw":1, "pw":0, "sl":[8,6], "inp":"IMAGE2", "out":"E-A-3",  "relu":0 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"CA1"  , "inc": 3  , "outc":32 , "sde":1,  "dw":0, "pw":1, "sl":[8,6], "inp":"E-A-3",  "out":"E-A-1",  "relu":1 ,"in_dim":192 }, # fnet.pointconv1
{ "name":"CE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A-1",  "out":"E-E",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"CF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":0, "sl":[5,6], "inp":"E-E",    "out":"E-F-3",  "relu":0 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"CF1"  , "inc": 16 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[5,6], "inp":"E-F-3",  "out":"E-F-1",  "relu":1 ,"in_dim":96  }, # fnet.layer2.0.pointconv2
{ "name":"CG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-F-1",  "out":"E-G",    "relu":1 ,"in_dim":96  }, # fnet.layer2.0.conv3
{ "name":"CP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-A-1",  "out":"E-EP1",  "relu":0 ,"in_dim":192  , "eop":[["load", "afm_2", "E-G"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer2.0.conv4
{ "name":"CH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-EP1",  "out":"E-H",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"CI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":0, "sl":[7,7], "inp":"E-H",    "out":"E-I-3",  "relu":0 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"CI1"  , "inc": 24 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[7,7], "inp":"E-I-3",  "out":"E-I-1",  "relu":1 ,"in_dim":48  }, # fnet.layer3.0.pointconv2
{ "name":"CJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-I-1",  "out":"E-J",    "relu":1 ,"in_dim":48  }, # fnet.layer3.0.conv3
{ "name":"CP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-EP1",  "out":"E-EP2",  "relu":0 ,"in_dim":96   , "eop":[["load", "afm_2", "E-J"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer3.0.conv4
{ "name":"CK1"  , "inc": 96 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,9], "inp":"E-EP2",  "out":"net",    "relu":0 ,"in_dim":48 ,  "eop":[["tanh", "afm_1", "afm_2", 0], "afm_2"]}, # fnet.conv2
{ "name":"CK2"  , "inc": 96 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,10],"inp":"E-EP2",  "out":"inp",    "relu":1 ,"in_dim":48  }, # fnet.conv2

#inp=net+2304*96, out=net+2304*160, flow=net+2304*240, q_pre=net+2304*242
{ "code":"A"    },
{ "iter":3      },
{ "code":"B"    },
{ "name":"A"    , "inc": 196, "outc":96 , "sde":1 , "dw":0, "pw":1, "sl":[0,7], "inp":"corr",   "out":"cor"   , "relu":1, "in_dim":48 }, # ud.encoder.convc1
{ "name":"B"    , "inc": 2  , "outc":64 , "sde":1 , "dw":0, "pw":1, "sl":[0,5], "inp":"FLOW",   "out":"flo1"  , "relu":1, "in_dim":48 }, # ud.encoder.convf1
{ "name":"C"    , "inc": 64 , "outc":32 , "sde":1 , "dw":0, "pw":1, "sl":[0,7], "inp":"flo1",   "out":"flo2"  , "relu":1, "in_dim":48 }, # ud.encoder.convf2
{ "name":"D3"   , "inc": 128, "outc":80 , "sde":1 , "dw":1, "pw":1, "sl":[6,7], "inp":"cor",    "out":"out"   , "relu":1, "in_dim":48 , "eop":[["flow", "afm_2", 0], "afm_2"]}, # ud.encoder.depthconv
{ "name":"F3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "sl":[6,5], "inp":"net",    "out":"z"     , "relu":0, "in_dim":48 , "eop":[["sigm", "afm_2", "afm_1"], "afm_1"]}, # ud.gru.depthconvz
{ "name":"E3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "sl":[6,5], "inp":"net",    "out":"q_pre" , "relu":0, "in_dim":48 , "eop":[["sigm", "afm_2", "afm_1"], ["load", "afm_3", "net"], ["mul", "afm_1", "afm_3", "afm_2", 6], "afm_2"]}, # ud.gru.depthconvr
{ "name":"G3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "sl":[5,7], "inp":"inp",    "out":"net"   , "relu":0, "in_dim":48 , "eop":[["tanh", "afm_2", "afm_1"], ["load", "afm_3", "net"], ["minus", "afm_1", "afm_3", "afm_2", 0], ["load", "afm_1", "z"], ["mul", "afm_1", "afm_2", "afm_4", 6], ["add", "afm_4", "afm_3", "afm_1", 0], "afm_1"]}, # ud.gru.depthconvq
{ "name":"H3"   , "inc": 96 , "outc":96 , "sde":1 , "dw":1, "pw":1, "sl":[6,8], "inp":"net",    "out":"f_head", "relu":1, "in_dim":48 }, # ud.flow_head.depthconv1
{ "name":"I"    , "inc": 96 , "outc":2  , "sde":1 , "dw":0, "pw":1, "sl":[0,7], "inp":"f_head", "out":"FLOW"  , "relu":0, "in_dim":48 }, # ud.flow_head.pointconv2
{ "iter":"end"  },

]

def demo(args):
    hlsGen = hlsGenerator(args)
    # hlsGen.getSingleConvText(2, 384, 384, 32, 32, True, True)
    # hlsGen.getSingleConvText(1, 192, 192, 32, 96, True, True)
    # hlsGen.getSingleConvText(1, 48, 48, 96, 96, True, True)
    fileWrapper(inText=hlsGen.getTextFromConfig(layerConfig, ddrConfig), fileTemplate=fileTemplate)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inParallel', help="input channel parallelism", default=32)
    parser.add_argument('--outParallel', help="output channel parallelism", default=32)
    parser.add_argument('--tileDims', help='on-chip buffer dims', default=48)
    args = parser.parse_args()

    demo(args)