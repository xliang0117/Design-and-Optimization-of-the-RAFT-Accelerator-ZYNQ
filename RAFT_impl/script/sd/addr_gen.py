import sys
import inspect
import numpy as np
import argparse
import colored_traceback
colored_traceback.add_hook()

layerConfig = [
{ "name":"EA3"  , "inc": 3  , "outc":32 , "sde":2,  "dw":1, "pw":1, "inp":"IMAGE      ", "out":"conv_0_out ", "relu":0 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"EP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "inp":"conv_5_out ", "out":"conv_16_out", "relu":0 ,"in_dim":192 }, # fnet.layer2.0.conv4
{ "name":"EE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "inp":"conv_5_out ", "out":"conv_12_out", "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"EF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":1, "inp":"conv_12_out", "out":"conv_13_out", "relu":0 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"EG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "inp":"conv_14_out", "out":"conv_15_out", "relu":1 ,"in_dim":96  , "eop":[["add", "afm_1", "afm_2", "afm_3", 1]]}, # fnet.layer2.0.conv3
{ "name":"EP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "inp":"conv_16_out", "out":"conv_26_out", "relu":0 ,"in_dim":96  }, # fnet.layer3.0.conv4
{ "name":"EH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "inp":"conv_16_out", "out":"conv_22_out", "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"EI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":1, "inp":"conv_22_out", "out":"conv_23_out", "relu":0 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"EJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "inp":"conv_24_out", "out":"conv_25_out", "relu":1 ,"in_dim":48 ,  "eop":[["add", "afm_1", "afm_2", "afm_3", 1]]}, # fnet.layer3.0.conv3
{ "name":"EK"   , "inc": 96 , "outc":96 , "sde":1,  "dw":0, "pw":1, "inp":"conv_30_out", "out":"conv_32_out", "relu":0 ,"in_dim":48}, # fnet.conv2


{ "name":"CA3"  , "inc": 3  , "outc":32 , "sde":2,  "dw":1, "pw":1, "inp":"IMAGE      ", "out":"conv_0_out ", "relu":0 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"CP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "inp":"conv_5_out ", "out":"conv_16_out", "relu":0 ,"in_dim":192 }, # fnet.layer2.0.conv4
{ "name":"CE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "inp":"conv_5_out ", "out":"conv_12_out", "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"CF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":1, "inp":"conv_12_out", "out":"conv_13_out", "relu":0 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"CG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "inp":"conv_14_out", "out":"conv_15_out", "relu":1 ,"in_dim":96  , "eop":[["add", "afm_1", "afm_2", "afm_3", 1]]}, # fnet.layer2.0.conv3
{ "name":"CP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "inp":"conv_16_out", "out":"conv_26_out", "relu":0 ,"in_dim":96  }, # fnet.layer3.0.conv4
{ "name":"CH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "inp":"conv_16_out", "out":"conv_22_out", "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"CI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":1, "inp":"conv_22_out", "out":"conv_23_out", "relu":0 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"CJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "inp":"conv_24_out", "out":"conv_25_out", "relu":1 ,"in_dim":48 ,  "eop":[["add", "afm_1", "afm_2", "afm_3", 1]]}, # fnet.layer3.0.conv3
{ "name":"CK1"  , "inc": 96 , "outc":96 , "sde":1,  "dw":0, "pw":1, "inp":"conv_30_out", "out":"conv_32_out", "relu":0 ,"in_dim":48}, # fnet.conv2
{ "name":"CK2"  , "inc": 96 , "outc":64 , "sde":1,  "dw":0, "pw":1, "inp":"conv_30_out", "out":"conv_32_out", "relu":0 ,"in_dim":48}, # fnet.conv2

#inp=net+2304*96, out=net+2304*160, flow=net+2304*240, q_pre=net+2304*242
{ "iter":3},
{ "name":"A"    , "inc": 196, "outc":96 , "sde":1 , "dw":0, "pw":1, "inp":"corr        ", "out":"cor           ", "relu":0, "in_dim":48 }, # ud.encoder.convc1
{ "name":"B"    , "inc": 2  , "outc":64 , "sde":1 , "dw":0, "pw":1, "inp":"net+2304*240", "out":"flo1          ", "relu":0, "in_dim":48 }, # ud.encoder.convf1
{ "name":"C"    , "inc": 64 , "outc":32 , "sde":1 , "dw":0, "pw":1, "inp":"flo1        ", "out":"cor+2304*96   ", "relu":0, "in_dim":48 }, # ud.encoder.convf2
{ "name":"D3"   , "inc": 128, "outc":80 , "sde":1 , "dw":1, "pw":1, "inp":"cor         ", "out":"conv_69_out   ", "relu":0, "in_dim":48 }, # ud.encoder.depthconv
{ "name":"F3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48 }, # ud.gru.depthconvz
{ "name":"E3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48 }, # ud.gru.depthconvr
{ "name":"G3"   , "inc": 242, "outc":96 , "sde":1 , "dw":1, "pw":1, "inp":"net+2304*96 ", "out":"update_dwout  ", "relu":0, "in_dim":48 }, # ud.gru.depthconvq
{ "name":"H3"   , "inc": 96 , "outc":96 , "sde":1 , "dw":1, "pw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48 }, # ud.flow_head.depthconv1
{ "name":"I3"   , "inc": 96 , "outc":2  , "sde":1 , "dw":0, "pw":1, "inp":"update_dwout", "out":"net+2304*240  ", "relu":0, "in_dim":48 }, # ud.flow_head.pointconv2
{ "iter":"end"},

]

ddrConfig = [
    # static
    {"name":"PRESERVE",     "ch":1,   "height":1,     "width": 385,   "start":None},

    # dynamic
    {"name":"E-A",          "ch":3,   "height":192,   "width": 192,   "start":None},
    {"name":"E-E",          "ch":16,  "height":192,   "width": 192,   "start":None},
    {"name":"E-F",          "ch":16,  "height":96,    "width": 96,    "start":None},
    {"name":"E-G",          "ch":64,  "height":96,    "width": 96,    "start":"E-E"},
    {"name":"E-EP1",        "ch":64,  "height":96,    "width": 96,    "start":None},
    {"name":"E-H",          "ch":24,  "height":96,    "width": 96,    "start":"E-A"},
    {"name":"E-I",          "ch":24,  "height":48,    "width": 48,    "start":None},
    {"name":"E-J",          "ch":96,  "height":48,    "width": 48,    "start":"E-H"},
    {"name":"E-EP2",        "ch":96,  "height":48,    "width": 48,    "start":None},

    {"name":"corr",         "ch":196, "height":48,    "width": 48,    "start":"E-A"},
    {"name":"cor",          "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"flo2",         "ch":32,  "height":48,    "width": 48,    "start":None},
    {"name":"flo1",         "ch":64,  "height":48,    "width": 48,    "start":None},
    {"name":"net",          "ch":96,  "height":48,    "width": 48,    "start":None},
    {"name":"inp",          "ch":64,  "height":48,    "width": 48,    "start":None},
    {"name":"out",          "ch":80,  "height":48,    "width": 48,    "start":None},
    {"name":"flow",         "ch":2,   "height":48,    "width": 48,    "start":None},
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

class addrGenerator():
    def __init__(self, ddr=ddrConfig, layer=layerConfig):
        self.ddr = ddr
        self.layer = layer
        self.maxDDRAddr = 0
        self.maxWeightAddr = 0
        self.ddrAddr = {}
        self.paramAddr = {}
        self.ddrAddrGen()
        self.weightAddrGen()

    def ddrAddrGen(self):
        self.ddrAddr["IMAGE1"] = "IMAGE1"
        self.ddrAddr["IMAGE2"] = "IMAGE2"
        self.ddrAddr["FLOW"] = "FLOW"
        currentAddr = 0
        maxAddr = 0
        for layer in self.ddr:
            if layer["start"] != None:
                if layer["start"] == "static":
                    currentAddr = maxAddr
                else:
                    currentAddr = self.ddrAddr[layer["start"]]
           
            self.ddrAddr[layer["name"]] = currentAddr

            channel_num = (layer["ch"] + 32 - 1) // 32
            currentAddr += channel_num * layer["height"] * layer["width"]
            maxAddr = currentAddr if currentAddr > maxAddr else maxAddr
        self.maxDDRAddr = maxAddr
    
    def weightAddrGen(self):
        currentAddr = 0
        for layer in self.layer:
            curLayer = layer.get("name", None)
            if curLayer:
                self.paramAddr[curLayer] = {}
                self.paramAddr[curLayer]["w3"] = currentAddr if layer["dw"] else 0
                currentAddr += ( 9*((layer["inc"]+32-1)//32) ) if layer["dw"] else 0

                self.paramAddr[curLayer]["dwb"] = currentAddr if layer["dw"] else 0
                currentAddr += ( (layer["inc"]+32-1)//32*2 ) if layer["dw"] else 0
                
                self.paramAddr[curLayer]["w1"] = currentAddr if layer["pw"] else 0
                currentAddr += ( ((layer["inc"]+32-1)//32*32) * ((layer["outc"]+32-1)//32) ) if layer["pw"] else 0

                self.paramAddr[curLayer]["pwb"] = currentAddr if layer["pw"] else 0
                currentAddr += ( (layer["outc"]+32-1)//32*2 ) if layer["pw"] else 0
        self.maxWeightAddr = currentAddr

    def getDDRAddr(self) -> dict:
        return self.ddrAddr

    def getWeightAddr(self) -> dict:
        return self.paramAddr

    def getMaxDDRAddr(self) -> int:
        return self.maxDDRAddr

    def getMaxWeightAddr(self) -> int:
        return self.maxWeightAddr


def demo(args):
    addrGen = addrGenerator(ddrConfig, layerConfig)
    print(addrGen.getDDRAddr())
    print(addrGen.getWeightAddr())



if __name__ == '__main__':
    demo(None)



    