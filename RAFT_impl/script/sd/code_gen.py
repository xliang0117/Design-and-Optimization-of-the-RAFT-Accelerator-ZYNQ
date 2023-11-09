import sys
import inspect
import numpy as np
import argparse
import colored_traceback
from addr_gen import addrGenerator
from ucodeTemplate import ucodeDict
colored_traceback.add_hook()

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
{ "split":0},
{ "name":"EA3"  , "inc": 3  , "outc":32 , "sde":2,  "dw":1, "pw":1, "sl":[9,5], "inp":"IMAGE1", "out":"E-A",    "relu":1 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"EE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A",    "out":"E-E",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"EF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":1, "sl":[5,7], "inp":"E-E",    "out":"E-F",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"EG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-F",    "out":"E-G",    "relu":1 ,"in_dim":96  }, # fnet.layer2.0.conv3
{ "name":"EP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A",    "out":"E-EP1",  "relu":0 ,"in_dim":192  , "eop":[["load", "afm_2", "E-G"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer2.0.conv4
{ "name":"EH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP1",  "out":"E-H",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"EI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":1, "sl":[6,7], "inp":"E-H",    "out":"E-I",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"EJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-I",    "out":"E-J",    "relu":1 ,"in_dim":48  }, # fnet.layer3.0.conv3
{ "name":"EP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP1",  "out":"E-EP2",  "relu":0 ,"in_dim":96   , "eop":[["load", "afm_2", "E-J"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer3.0.conv4
{ "name":"EK"   , "inc": 96 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-EP2",  "out":"E-fmap1","relu":0 ,"in_dim":48  }, # fnet.conv2


{ "name":"CA3"  , "inc": 3  , "outc":32 , "sde":2,  "dw":1, "pw":1, "sl":[8,6], "inp":"IMAGE2", "out":"E-A",    "relu":1 ,"in_dim":384 }, # fnet.depthconv1
{ "name":"CE"   , "inc": 32 , "outc":16 , "sde":1,  "dw":0, "pw":1, "sl":[0,7], "inp":"E-A",    "out":"E-E",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.conv1
{ "name":"CF3"  , "inc": 16 , "outc":16 , "sde":2,  "dw":1, "pw":1, "sl":[5,6], "inp":"E-E",    "out":"E-F",    "relu":1 ,"in_dim":192 }, # fnet.layer2.0.depthconv2
{ "name":"CG"   , "inc": 16 , "outc":64 , "sde":1,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-F",    "out":"E-G",    "relu":1 ,"in_dim":96  }, # fnet.layer2.0.conv3
{ "name":"CP1"  , "inc": 32 , "outc":64 , "sde":2,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-A",    "out":"E-EP1",  "relu":0 ,"in_dim":192  , "eop":[["load", "afm_2", "E-G"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer2.0.conv4
{ "name":"CH"   , "inc": 64 , "outc":24 , "sde":1,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-EP1",  "out":"E-H",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.conv1
{ "name":"CI3"  , "inc": 24 , "outc":24 , "sde":2,  "dw":1, "pw":1, "sl":[7,7], "inp":"E-H",    "out":"E-I",    "relu":1 ,"in_dim":96  }, # fnet.layer3.0.depthconv2
{ "name":"CJ"   , "inc": 24 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,5], "inp":"E-I",    "out":"E-J",    "relu":1 ,"in_dim":48  }, # fnet.layer3.0.conv3
{ "name":"CP2"  , "inc": 64 , "outc":96 , "sde":2,  "dw":0, "pw":1, "sl":[0,6], "inp":"E-EP1",  "out":"E-EP2",  "relu":0 ,"in_dim":96   , "eop":[["load", "afm_2", "E-J"], ["add", "afm_1", "afm_2", "afm_3", 1], "afm_3"]}, # fnet.layer3.0.conv4
{ "name":"CK1"  , "inc": 96 , "outc":96 , "sde":1,  "dw":0, "pw":1, "sl":[0,9], "inp":"E-EP2",  "out":"net",    "relu":0 ,"in_dim":48 ,  "eop":[["tanh", "afm_1", "afm_2"], "afm_2"]}, # fnet.conv2
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

class hlsGenerator:
    def __init__(self, args:dict) -> None:
        self.inParallel = args.inParallel
        self.outParallel = args.outParallel
        self.tileDims = args.tileDims

    def get_kwargs(self):
        frame = inspect.currentframe().f_back
        keys, _, _, values = inspect.getargvalues(frame)
        kwargs = {}
        for key in keys:
            if key != 'self':
                kwargs[key] = values[key]
        return kwargs

    def loadFMText(self, stride=1, full=True, dims=32, fm="afm_1", colx="colx", rowx="rowx", chx="inx", addr=0)-> str:
        keyDict = self.get_kwargs()
        if addr in ["IMAGE1", "IMAGE2"]:
            return "load_fm_IMAGE({addr}, {fm}, {rowx}, {colx});\n".format(**keyDict)
        elif addr == "FLOW":
            return "load_fm_flow(flow, {fm});\n".format(**keyDict)
        elif stride > 1:
            return "load_fm_s2(fm + {addr}, {fm}, {rowx}, {colx}, {chx}, {dims}, {dims});\n".format(**keyDict)
        elif full:
            keyDict["addr"] = keyDict["addr"] - 48 - 1 #for padding
            return "load_fm_full(fm + {addr}, {fm}, {chx});\n".format(**keyDict)
        else:
            keyDict["addr"] = keyDict["addr"] - dims - 1 #for padding
            return "load_fm_tile(fm + {addr}, {fm}, {rowx}, {colx}, {chx}, {dims}, {dims});\n".format(**keyDict)

    def exportFMText(self, stride=1, full=True, dims=32, fm="afm_2", colx="colx", rowx="rowx", chx="chx", addr=0)-> str:
        keyDict = self.get_kwargs()
        if addr == "FLOW":
            return "export_flow({fm}, flow);\n".format(**keyDict)
        elif stride > 1:
            return "export_fm_s2(fm + {addr}, {fm}, {rowx}, {colx}, {chx}, {dims}, {dims});\n".format(**keyDict)
        elif full:
            return "export_fm_full(fm + {addr}, {fm}, {chx});\n".format(**keyDict)
        else:
            return "export_fm_tile(fm + {addr}, {fm}, {rowx}, {colx}, {chx}, {dims}, {dims});\n".format(**keyDict)

    def loadParamText(self, op="w1", inx=0, outPart=1, pingPong="A", addr=0) -> str:
        keyDict = self.get_kwargs()
        if op == "w1":
            return "load_w1x1(weight + {addr}, wBUF1x1_{pingPong}, {inx}, {outPart});\n".format(**keyDict)
        elif op == "w3":
            return "load_w3x3(weight + {addr}, wBUF3x3_{pingPong}, {inx});\n".format(**keyDict)
        elif op == "dwb":
            return "load_dwbbuf(weight + {addr}, dwb_buf_{pingPong}, {inx});\n".format(**keyDict)
        elif op == "pwb":
            return "load_pwbbuf(weight + {addr}, pwb_buf, {outPart});\n".format(**keyDict)

    def dwConvText(self, inFM="afm_1", outFM="afm_2", pingPong="A", scale=0) -> str:
        keyDict = self.get_kwargs()
        return "dw_conv_3x3({inFM}, {outFM}, wBUF3x3_{pingPong}, dwb_buf_{pingPong}, {scale});\n".format(**keyDict)

    def pwConvText(self, inFM="afm_1", pingPong="A", outx="outx") -> str:
        keyDict = self.get_kwargs()
        return "pw_conv_1x1({inFM}, rfm[{outx}], wBUF1x1_{pingPong}[{outx}]);\n".format(**keyDict)

    def pwConvGroupText(self, inFM="afm_1", pingPong="A", outPart=3) -> str:
        keyDict = self.get_kwargs()
        return "pw_conv_group({inFM}, rfm, wBUF1x1_{pingPong}, {outPart});\n".format(**keyDict)

    def actText(self, outFM="afm_2", outx="outx", scale=0, relu="ture") -> str:
        keyDict = self.get_kwargs()
        return "activation(rfm[{outx}], {outFM}, pwb_buf[{outx}], {scale}, {relu});\n".format(**keyDict)

    def elemText(self, op="add", ifm1="afm_1", ifm2="afm_2", ofm="afm_3", scale=0, relu="true"):
        keyDict = self.get_kwargs()
        if op == "add":
            return "res_add_sub({ifm1}, {ifm2}, {ofm}, true, {relu});\n".format(**keyDict)
        elif op == "minus":
            return "res_add_sub({ifm1}, {ifm2}, {ofm}, false, {relu});\n".format(**keyDict)
        elif op == "mul":
            return "res_mul({ifm1}, {ifm2}, {ofm}, {scale});\n".format(**keyDict)
        elif op == "tanh":
            return "tanh_by_point({ifm1}, {ofm});\n".format(**keyDict)
        elif op == "sigm":
            return "sigmoid_by_point({ifm1}, {ofm});\n".format(**keyDict)
        elif op == "flow":
            return "out_with_flow({ifm1}, flow, outx);\n".format(**keyDict)

    def eopText(self, eop=[["add", "afm_1", "afm_2", "afm_3", 1]], full=True, outDims=0, chx="outx", tabs=1, addr=0):
        text = ""
        for singleOP in eop[0:-1]:
            if singleOP[0] == "load":
                text += " "*tabs*4 + self.loadFMText(stride=1, fm=singleOP[1], full=full, dims=outDims, chx=chx, addr=singleOP[2])
            elif singleOP[0] == "flow":
                text += " "*tabs*4 + self.elemText(op=singleOP[0], ifm1=singleOP[1])
            elif singleOP[0] in ["tanh", "sigm"]:
                text += " "*tabs*4  + self.elemText(op=singleOP[0], ifm1=singleOP[1], ofm=singleOP[2])
            else:
                reluT = "true" if singleOP[4] else "false"
                text += " "*tabs*4  + self.elemText(op=singleOP[0], ifm1=singleOP[1], ifm2=singleOP[2], ofm=singleOP[3], scale=singleOP[4], relu=reluT)
        return text

    def iterText(self, iter=3):
        iterT = "\nfor(short iter = 0; iter < {}; iter++) {{\n".format(iter) if iter != "end" else "}"
        return iterT
    
    def ucodeText(self, code="A", addr={}):
        return ucodeDict[code].format(**addr)


    def onlyRowColSeqConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM) -> str:

        return  (" "*4  +   self.loadParamText("w1", outPart=1, addr=w1Addr) if pw else "") + \
                (" "*4  +   self.loadParamText("w3", inx=0, outPart=0, pingPong="A", addr=w3Addr) if dw else "") + \
                (" "*4  +   self.loadParamText("dwb", inx=0, outPart=0, pingPong="A", addr=dwbAddr) if dw else "") + \
                (" "*4  +   self.loadParamText("pwb", inx=0, outPart=1, pingPong="A", addr=pwbAddr) if pw else "")+ \
                 " "*4  +   "{}_R:for (short rowx = 0; rowx < {}; rowx++) {{\n".format(comment, rowPart) + \
                 " "*8  +       "for (short colx = 0; colx < {}; colx++) {{\n".format(colPart) + \
                 " "*12 +           self.loadFMText(ldStride, full, rowDims, chx=0, addr=inAddr) + \
                (" "*12 +           self.dwConvText(dwFM, pwFM, pingPong="A", scale=dwS) if dw else "") + \
                (" "*12 +           self.pwConvText(pwFM, pingPong="A", outx=0) if pw else "") + \
                (" "*12 +           self.actText(pwFM, outx=0, scale=pwS, relu=reluT) if pw else "")+ \
                 " "*0  +           self.eopText(eop, full, outDims, chx=0, tabs=3) + \
                 " "*12 +           self.exportFMText(exStride, full, outDims, eopFM, chx=0, addr=outAddr) + \
                 " "*8  +       "}\n" +\
                 " "*4  +   "}\n"


    def onlyCinCoutSeqConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   "{}_I:for (short inx = 0; inx < {}; inx++) {{\n".format(comment,inPart) + \
                 " "*8  +       self.loadFMText(ldStride, full, rowDims, colx=0, rowx=0, chx="inx", addr=inAddr) + \
                (" "*8  +       self.loadParamText("w3", inx="inx", pingPong="A", addr=w3Addr) if dw else "") + \
                 " "*8  +       self.loadParamText("w1", inx="inx", outPart=outPart, addr=w1Addr) + \
                (" "*8  +       self.loadParamText("dwb", inx="inx", addr=dwbAddr) if dw else "") + \
                (" "*8  +       self.dwConvText(dwFM, pwFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*8  +       "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                 " "*12 +           self.pwConvText(pwFM, pingPong="A", outx="outx") + \
                 " "*8  +       "}\n" + \
                 " "*4  +   "}\n" + \
                 " "*4  +   "{}_O:for (short outx = 0; outx < {}; outx++) {{\n".format(comment, outPart) + \
                 " "*8  +       self.actText(pwFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +       self.eopText(eop, full, outDims, tabs=2) + \
                 " "*8  +       self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*4  +   "}\n"


    def allRowColCinCoutSeqConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   "{}_R:for (short rowx = 0; rowx < {}; rowx++) {{\n".format(comment, rowPart) + \
                 " "*8  +       "for (short colx = 0; colx < {}; colx++) {{\n".format(colPart) + \
                 " "*12 +           "for (short inx = 0; inx < {}; inx++) {{\n".format(inPart) + \
                 " "*16 +               self.loadFMText(ldStride, full, rowDims, chx="inx", addr=inAddr) + \
                (" "*16 +               self.loadParamText("w3", inx="inx", pingPong="A", addr=w3Addr) if dw else "") + \
                 " "*16 +               self.loadParamText("w1", inx="inx", outPart=outPart, addr=w1Addr) + \
                (" "*16 +               self.loadParamText("dwb", inx="inx", addr=dwbAddr) if dw else "") + \
                (" "*16 +               self.dwConvText(dwFM, pwFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*16 +               "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                 " "*20 +                   self.pwConvText(pwFM, pingPong="A", outx="outx") + \
                 " "*16 +               "}\n" + \
                 " "*12 +           "}\n" + \
                 " "*12 +           "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                 " "*16 +               self.actText(pwFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +               self.eopText(eop, full, outDims, tabs=4) + \
                 " "*16 +               self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*12 +           "}\n" + \
                 " "*8  +       "}\n" + \
                 " "*4  +   "}\n"

    def allRowColCinCoutSeqGroupConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   "{}_R:for (short rowx = 0; rowx < {}; rowx++) {{\n".format(comment, rowPart) + \
                 " "*8  +       "for (short colx = 0; colx < {}; colx++) {{\n".format(colPart) + \
                 " "*12 +           "for (short inx = 0; inx < {}; inx++) {{\n".format(inPart) + \
                 " "*16 +               self.loadFMText(ldStride, full, rowDims, chx="inx", addr=inAddr) + \
                (" "*16 +               self.loadParamText("w3", inx="inx", pingPong="A", addr=w3Addr) if dw else "") + \
                 " "*16 +               self.loadParamText("w1", inx="inx", outPart=outPart, addr=w1Addr) + \
                (" "*16 +               self.loadParamText("dwb", inx="inx", addr=dwbAddr) if dw else "") + \
                (" "*16 +               self.dwConvText(dwFM, pwFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*16 +               self.pwConvGroupText(pwFM, pingPong="A", outPart=outPart) + \
                 " "*12 +           "}\n" + \
                 " "*12 +           "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                 " "*16 +               self.actText(pwFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +               self.eopText(eop, full, outDims, tabs=4) + \
                 " "*16 +               self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*12 +           "}\n" + \
                 " "*8  +       "}\n" + \
                 " "*4  +   "}\n"


    def onlyRowColCoutSheduleConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, eopPingFM, eopPongFM) -> str:

        return  (" "*4  +   self.loadParamText("w1", outPart=outPart, addr=w1Addr) if pw else "") + \
                (" "*4  +   self.loadParamText("w3", inx=0, outPart=outPart, pingPong="A", addr=w3Addr) if dw else "") + \
                (" "*4  +   self.loadParamText("dwb", inx=0, outPart=outPart, pingPong="A", addr=dwbAddr) if dw else "") + \
                (" "*4  +   self.loadParamText("pwb", inx=0, outPart=outPart, pingPong="A", addr=pwbAddr) if pw else "")+ \
                 " "*4  +   "{}_R:for (short rowx = 0; rowx < {}; rowx++) {{\n".format(comment, rowPart) + \
                 " "*8  +       self.loadFMText(ldStride, full, rowDims, fm=dwPingFM, colx=0, chx=0, addr=inAddr) + \
                 " "*8  +       "for (short colx = 0; colx < {}; colx++) {{\n".format(colPart) + \
                 " "*12  +           "if(colx % 2 == 0) {\n" + \
                 " "*16 +               self.loadFMText(ldStride, full, rowDims, fm=dwPongFM, colx="colx+1", chx=0, addr=inAddr) + \
                (" "*16 +               self.dwConvText(dwPingFM, pwPingFM, pingPong="A", scale=dwS) if dw else "") + \
                (" "*16 +               self.pwConvGroupText(pwPingFM, pingPong="A", outPart=outPart) if pw else "") + \
                 " "*16  +              "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                (" "*20 +                   self.actText(pwPingFM, scale=pwS, relu=reluT) if pw else "")+ \
                 " "*0  +                   self.eopText(eop, full, outDims, chx=0, tabs=3) + \
                 " "*20 +                   self.exportFMText(exStride, full, outDims, eopPingFM, chx="outx", addr=outAddr) + \
                 " "*16 +               "}\n" + \
                 " "*12 +           "}\n" + \
                 " "*12 +           "else {\n" + \
                 " "*16 +               self.loadFMText(ldStride, full, rowDims, fm=dwPingFM, colx="colx+1", chx=0, addr=inAddr) + \
                (" "*16 +               self.dwConvText(dwPongFM, pwPongFM, pingPong="A", scale=dwS) if dw else "") + \
                (" "*16 +               self.pwConvGroupText(pwPongFM, pingPong="A", outPart=outPart) if pw else "") + \
                 " "*16  +              "for (short outx = 0; outx < {}; outx++) {{\n".format(outPart) + \
                (" "*20 +                   self.actText(pwPongFM, scale=pwS, relu=reluT) if pw else "")+ \
                 " "*0  +                   self.eopText(eop, full, outDims, chx=0, tabs=3) + \
                 " "*20 +                   self.exportFMText(exStride, full, outDims, eopPongFM, chx="outx", addr=outAddr) + \
                 " "*16 +               "}\n" + \
                 " "*12 +           "}\n" + \
                 " "*8  +       "}\n" +\
                 " "*4  +   "}\n"


    def onlyCinCoutSheduleConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, 
                                full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   self.loadFMText(ldStride, full, rowDims, chx=0, colx=0, rowx=0, fm=dwPingFM, addr=inAddr) + \
                (" "*4  +   self.loadParamText("w3", inx=0, pingPong="A", addr=w3Addr)if dw else "") + \
                 " "*4  +   self.loadParamText("w1", inx=0, pingPong="A", outPart=outPart, addr=w1Addr) + \
                (" "*4  +   self.loadParamText("dwb", inx=0, pingPong="A", addr=dwbAddr)if dw else "") + \
                 " "*4  +   "{}_I:for (short inx = 0; inx < {}; inx++) {{\n".format(comment,inPart) + \
                 " "*8  +       "if(inx % 2 == 0) {\n" + \
                 " "*12 +           self.loadFMText(ldStride, full, rowDims, chx="inx+1", colx=0, rowx=0, fm=dwPongFM, addr=inAddr) + \
                (" "*12 +           self.loadParamText("w3", inx="inx+1", pingPong="B", addr=w3Addr)if dw else "") + \
                 " "*12 +           self.loadParamText("w1", inx="inx+1", pingPong="B", outPart=outPart, addr=w1Addr) + \
                (" "*12 +           self.loadParamText("dwb", inx="inx+1", pingPong="B", addr=dwbAddr)if dw else "") + \
                (" "*12 +           self.dwConvText(dwPingFM, pwPingFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*12 +           self.pwConvGroupText(pwPingFM, pingPong="A", outPart=outPart) + \
                 " "*8  +       "}\n" + \
                 " "*8  +       "else {\n" + \
                 " "*12 +           self.loadFMText(ldStride, full, rowDims, chx="inx+1", colx=0, rowx=0, fm=dwPingFM, addr=inAddr) + \
                (" "*12 +           self.loadParamText("w3", inx="inx+1", pingPong="A", addr=w3Addr)if dw else "") + \
                 " "*12 +           self.loadParamText("w1", inx="inx+1", pingPong="A", outPart=outPart, addr=w1Addr) + \
                (" "*12 +           self.loadParamText("dwb", inx="inx+1", pingPong="A", addr=dwbAddr)if dw else "") + \
                (" "*12 +           self.dwConvText(dwPongFM, pwPongFM, pingPong="B", scale=dwS) if dw else "") + \
                 " "*12 +           self.pwConvGroupText(pwPongFM, pingPong="B", outPart=outPart) + \
                 " "*8  +       "}\n" + \
                 " "*4  +   "}\n" + \
                 " "*4  +   "{}_O:for (short outx = 0; outx < {}; outx++) {{\n".format(comment, outPart) + \
                 " "*8  +       self.actText(pwPingFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +       self.eopText(eop, full, outDims, tabs=2) + \
                 " "*8  +       self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*4  +   "}\n"


    def onlyCoutSheduleConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   self.loadFMText(ldStride, full, chx=0, colx=0, rowx=0, fm=dwFM, addr=inAddr) + \
                (" "*4  +   self.loadParamText("w3", inx=0, pingPong="A", addr=w3Addr)if dw else "") + \
                 " "*4  +   self.loadParamText("w1", inx=0, pingPong="A", outPart=outPart, addr=w1Addr) + \
                (" "*4  +   self.loadParamText("dwb", inx=0, pingPong="A", addr=dwbAddr)if dw else "") + \
                (" "*4  +   self.dwConvText(dwFM, pwFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*4  +   self.pwConvGroupText(pwFM, pingPong="A", outPart=outPart) + \
                 " "*4  +   "{}_O:for (short outx = 0; outx < {}; outx++) {{\n".format(comment, outPart) + \
                 " "*8  +       self.actText(pwFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +       self.eopText(eop, full, outDims, tabs=2) + \
                 " "*8  +       self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*4  +   "}\n"


    def allRowColCinCoutSheduleConvText(self, w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, eopFM) -> str:

        return   " "*4  +   self.loadParamText("pwb", outPart=outPart, addr=pwbAddr) + \
                 " "*4  +   "{}_R:for (short rowx = 0; rowx < {}; rowx++) {{\n".format(comment, rowPart) + \
                 " "*8  +       "for (short colx = 0; colx < {}; colx++) {{\n".format(colPart) + \
                 " "*12 +           self.loadFMText(ldStride, full, rowDims, chx=0, fm=dwPingFM, addr=inAddr) + \
                (" "*12 +           self.loadParamText("w3", inx=0, pingPong="A", addr=w3Addr)if dw else "") + \
                 " "*12 +           self.loadParamText("w1", inx=0, pingPong="A", outPart=outPart, addr=w1Addr) + \
                (" "*12 +           self.loadParamText("dwb", inx=0, pingPong="A", addr=dwbAddr)if dw else "") + \
                 " "*12 +           "{}_I:for (short inx = 0; inx < {}; inx++) {{\n".format(comment,inPart) + \
                 " "*16 +               "if(inx % 2 == 0) {\n" + \
                 " "*20 +                   self.loadFMText(ldStride, full, rowDims, chx="inx+1", fm=dwPongFM, addr=inAddr) + \
                (" "*20 +                   self.loadParamText("w3", inx="inx+1", pingPong="B", addr=w3Addr)if dw else "") + \
                 " "*20 +                   self.loadParamText("w1", inx="inx+1", pingPong="B", outPart=outPart, addr=w1Addr) + \
                (" "*20 +                   self.loadParamText("dwb", inx="inx+1", pingPong="B", addr=dwbAddr)if dw else "") + \
                (" "*20 +                   self.dwConvText(dwPingFM, pwPingFM, pingPong="A", scale=dwS) if dw else "") + \
                 " "*20 +                   self.pwConvGroupText(pwPingFM, pingPong="A", outPart=outPart) + \
                 " "*16 +               "}\n" + \
                 " "*16 +               "else {\n" + \
                 " "*20 +                   self.loadFMText(ldStride, full, rowDims, chx="inx+1", fm=dwPingFM, addr=inAddr) + \
                (" "*20 +                   self.loadParamText("w3", inx="inx+1", pingPong="A", addr=w3Addr)if dw else "") + \
                 " "*20 +                   self.loadParamText("w1", inx="inx+1", pingPong="A", outPart=outPart, addr=w1Addr) + \
                (" "*20 +                   self.loadParamText("dwb", inx="inx+1", pingPong="A", addr=dwbAddr)if dw else "") + \
                (" "*20 +                   self.dwConvText(dwPongFM, pwPongFM, pingPong="B", scale=dwS) if dw else "") + \
                 " "*20 +                   self.pwConvGroupText(pwPongFM, pingPong="B", outPart=outPart) + \
                 " "*16 +               "}\n" + \
                 " "*12 +           "}\n" + \
                 " "*12 +           "{}_O:for (short outx = 0; outx < {}; outx++) {{\n".format(comment, outPart) + \
                 " "*16 +               self.actText(pwPingFM, outx="outx", scale=pwS, relu=reluT) + \
                 " "*0  +               self.eopText(eop, full, outDims, tabs=2) + \
                 " "*16 +               self.exportFMText(exStride, full, outDims, eopFM, chx="outx", addr=outAddr) + \
                 " "*12 +           "}\n" + \
                 " "*8  +       "}\n" + \
                 " "*4  +   "}\n"
    

    def seqConvCodeGen(self, comment="", stride=1, rowDims=48, colDims=48, inChannel=32, outChannel=32, dw=0, pw=0, dwS=0, pwS=0, relu=0, eop=[],
                    w3Addr=0, w1Addr=0, dwbAddr=0, pwbAddr=0, inAddr=0, outAddr=0, split=0) -> str:
        rowPart = (rowDims + self.tileDims - 1) // self.tileDims
        colPart = (colDims + self.tileDims - 1) // self.tileDims
        inPart = (inChannel + self.inParallel - 1) // self.inParallel
        outPart = (outChannel + self.outParallel - 1) // self.outParallel

        pw = pw if split else 1
        reluT = "true" if relu else "false"
        ldStride = 1 if dw else stride
        exStride = stride if dw else 1
        outDims = rowDims // ldStride // exStride
        rowPart, colPart = rowPart // ldStride, colPart//ldStride
        full = (rowPart == colPart == 1)

        dwFM = "afm_1"
        pwFM = "afm_2" if dw else "afm_1"
        eopFM = eop[-1] if eop else pwFM
        commentText = "\n//{cmt}, {idims}*{idims}*{inc} -> {odims}*{odims}*{outc}\n".format(cmt=comment, idims=rowDims, inc=inChannel, odims=rowDims//stride, outc=outChannel)

        if inPart == outPart == 1:
            cvTxt =  self.onlyRowColSeqConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM)
        elif rowPart == colPart == 1:
            cvTxt =  self.onlyCinCoutSeqConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM)
        else:
            cvTxt = self.allRowColCinCoutSeqConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM)
        return commentText + cvTxt


    def sheduleConvCodeGen(self, comment="", stride=1, rowDims=48, colDims=48, inChannel=32, outChannel=32, dw=0, pw=0, dwS=0, pwS=0, relu=0, eop=[],
                    w3Addr=0, w1Addr=0, dwbAddr=0, pwbAddr=0, inAddr=0, outAddr=0, split=0) -> str:
        rowPart = (rowDims + self.tileDims - 1) // self.tileDims
        colPart = (colDims + self.tileDims - 1) // self.tileDims
        inPart = (inChannel + self.inParallel - 1) // self.inParallel
        outPart = (outChannel + self.outParallel - 1) // self.outParallel

        pw = pw if split else 1
        reluT = "true" if relu else "false"
        ldStride = 1 if dw else stride
        exStride = stride if dw else 1
        outDims = rowDims // ldStride // exStride
        rowPart, colPart = rowPart // ldStride, colPart//ldStride
        full = (rowPart == colPart == 1)

        dwFM = "afm_1"
        pwFM = "afm_2" if dw else "afm_1"
        eopFM = eop[-1] if eop else pwFM
        if dw:
            dwPingFM, dwPongFM, pwPingFM, pwPongFM = "afm_1", "afm_3", "afm_2", "afm_4"
        else:
            dwPingFM, dwPongFM, pwPingFM, pwPongFM = "afm_1", "afm_3", "afm_1", "afm_3"
        commentText = "\n//{cmt}, {idims}*{idims}*{inc} -> {odims}*{odims}*{outc}\n".format(cmt=comment, idims=rowDims, inc=inChannel, odims=rowDims//stride, outc=outChannel)

        if rowPart == colPart == inPart == 1:
            cvTxt = self.onlyCoutSheduleConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM)
        elif rowPart == colPart == 1:
            cvTxt = self.onlyCinCoutSheduleConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, inPart, outPart, ldStride, exStride, 
                                full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, eopFM)
        elif inPart == 1 and not eop:
            cvTxt = self.onlyRowColCoutSheduleConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, pwPingFM, pwPongFM)
        elif inPart == 1:
            cvTxt = self.allRowColCinCoutSeqGroupConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwFM, pwFM, reluT, eop, eopFM)
        else:
            cvTxt = self.allRowColCinCoutSheduleConvText(w1Addr, w3Addr, dwbAddr, pwbAddr, inAddr, outAddr, 
                                dw, pw, dwS, pwS, comment, rowPart, colPart, inPart, outPart, 
                                ldStride, exStride, full, rowDims, outDims, dwPingFM, dwPongFM, pwPingFM, pwPongFM, reluT, eop, eopFM)
        return commentText + cvTxt


    def getSingleConvText(self, comment="", stride=1, rowDims=48, colDims=48, inChannel=32, outChannel=32, dw=False, relu=False):
        print(self.convCodeGen(comment, stride, rowDims, colDims, inChannel, outChannel, dw, relu))


    def getTextFromConfig(self, layerConfig:list, ddrConfig:list, split=0) -> str:
        addr = addrGenerator(ddr=ddrConfig, layer=layerConfig)
        fmDDR = addr.getDDRAddr()
        weightDDR = addr.getWeightAddr()
        print("DDRAddr:{}".format(addr.ddrAddr))
        print("maxDDRAddr:{}".format(addr.getMaxDDRAddr()))
        print("maxWeightAddr:{}".format(addr.getMaxWeightAddr()))
        uConfig = layerConfig[0]
        sheduled = uConfig.get("sheduled", None)
        allText = ""
        for sc in layerConfig[1::]:
            if not isinstance(sc, dict):
                raise TypeError("Config not contain dict")
            eop = sc.get("eop", [])
            for singleEOP in eop:
                if singleEOP[0]=="load":
                    singleEOP[2] = fmDDR[singleEOP[2]]
            dwS, pwS = sc.get("sl", [0,0])
            iter = sc.get("iter", None)
            code = sc.get("code", None)

            if iter:
                allText += self.iterText(iter)
            elif code:
                allText += self.ucodeText(code, fmDDR)
            elif sheduled:
                allText += self.sheduleConvCodeGen(comment=sc["name"], stride=sc["sde"], rowDims=sc["in_dim"], colDims=sc["in_dim"], inChannel=sc["inc"], 
                                        outChannel=sc["outc"], dw=sc["dw"], pw=sc["pw"], dwS=dwS, pwS=pwS, relu=sc["relu"], eop=eop,
                                        dwbAddr=weightDDR[sc["name"]]["dwb"], pwbAddr=weightDDR[sc["name"]]["pwb"], w3Addr=weightDDR[sc["name"]]["w3"], 
                                        w1Addr=weightDDR[sc["name"]]["w1"],inAddr=fmDDR[sc["inp"]], outAddr=fmDDR[sc["out"]], split=uConfig["split"])
            else:
                allText += self.seqConvCodeGen(comment=sc["name"], stride=sc["sde"], rowDims=sc["in_dim"], colDims=sc["in_dim"], inChannel=sc["inc"], 
                                        outChannel=sc["outc"], dw=sc["dw"], pw=sc["pw"], dwS=dwS, pwS=pwS, relu=sc["relu"], eop=eop,
                                        dwbAddr=weightDDR[sc["name"]]["dwb"], pwbAddr=weightDDR[sc["name"]]["pwb"], w3Addr=weightDDR[sc["name"]]["w3"], 
                                        w1Addr=weightDDR[sc["name"]]["w1"], inAddr=fmDDR[sc["inp"]], outAddr=fmDDR[sc["out"]], split=uConfig["split"])
        return allText

def demo(args):
    hlsGen = hlsGenerator(args)
    # hlsGen.getSingleConvText(2, 384, 384, 32, 32, True, True)
    # hlsGen.getSingleConvText(1, 192, 192, 32, 96, True, True)
    # hlsGen.getSingleConvText(1, 48, 48, 96, 96, True, True)
    print(hlsGen.getTextFromConfig(layerConfig, ddrConfig))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inParallel', help="input channel parallelism", default=32)
    parser.add_argument('--outParallel', help="output channel parallelism", default=32)
    parser.add_argument('--tileDims', help='on-chip buffer dims', default=48)
    parser.add_argument('--split', help='split 3x3', default=0)
    args = parser.parse_args()

    demo(args)

    

