import sys
sys.path.append('core')
sys.path.append('gemmini')

import numpy as np
import argparse
import torch
from raft import RAFT
from gemmini.gemminiText import *

layer_config = [
{ "name":"conv_0 ", "inc": 3  , "outc":3  , "ks":3  , "pad":1  , "sde":2  , "dw":1, "inp":"IMAGE      ", "out":"conv_0_out ", "relu":0 ,"in_dim":384, "pt":"fe"}, # fnet.depthconv1
{ "name":"conv_1 ", "inc": 3  , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_0_out ", "out":"conv_1_out ", "relu":1 ,"in_dim":192, "pt":"fe"}, # fnet.pointconv1

{ "name":"conv_2 ", "inc": 32 , "outc":8  , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_1_out ", "out":"conv_2_out ", "relu":1 ,"in_dim":192, "pt":"fe"}, # fnet.layer1.0.conv1
{ "name":"conv_3 ", "inc": 8  , "outc":8  , "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"conv_2_out ", "out":"conv_3_out ", "relu":0 ,"in_dim":192, "pt":"fe"}, # fnet.layer1.0.depthconv2
{ "name":"conv_4 ", "inc": 8  , "outc":8  , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_3_out ", "out":"conv_4_out ", "relu":1 ,"in_dim":192, "pt":"fe"}, # fnet.layer1.0.pointconv2
{ "name":"conv_5 ", "inc": 8  , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_4_out ", "out":"conv_5_out ", "relu":1 ,"in_dim":192, "pt":"fe"}, # fnet.layer1.0.conv3
{ "name":"res_add", "inc": 32 , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_1_out ", "out":"conv_5_out ", "relu":1 ,"in_dim":192, "pt":"fe", "cfg":"conv_5"}, # res_add1

{ "name":"conv_12", "inc": 32 , "outc":16 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_5_out ", "out":"conv_12_out", "relu":1 ,"in_dim":192, "pt":"fe"}, # fnet.layer2.0.conv1
{ "name":"conv_13", "inc": 16 , "outc":16 , "ks":3  , "pad":1  , "sde":2  , "dw":1, "inp":"conv_12_out", "out":"conv_13_out", "relu":0 ,"in_dim":192, "pt":"fe"}, # fnet.layer2.0.depthconv2
{ "name":"conv_14", "inc": 16 , "outc":16 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_13_out", "out":"conv_14_out", "relu":1 ,"in_dim":96 , "pt":"fe"}, # fnet.layer2.0.pointconv2
{ "name":"conv_15", "inc": 16 , "outc":64 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_14_out", "out":"conv_15_out", "relu":1 ,"in_dim":96 , "pt":"fe"}, # fnet.layer2.0.conv3
{ "name":"conv_16", "inc": 32 , "outc":64 , "ks":1  , "pad":0  , "sde":2  , "dw":0, "inp":"conv_5_out ", "out":"conv_16_out", "relu":0 ,"in_dim":192, "pt":"fe"}, # fnet.layer2.0.conv4
{ "name":"res_add", "inc": 32 , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_15_out", "out":"conv_16_out", "relu":1 ,"in_dim":96 , "pt":"fe", "cfg":"conv_16"}, # res_add2

{ "name":"conv_22", "inc": 64 , "outc":24 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_16_out", "out":"conv_22_out", "relu":1 ,"in_dim":96 , "pt":"fe"}, # fnet.layer3.0.conv1
{ "name":"conv_23", "inc": 24 , "outc":24 , "ks":3  , "pad":1  , "sde":2  , "dw":1, "inp":"conv_22_out", "out":"conv_23_out", "relu":0 ,"in_dim":96 , "pt":"fe"}, # fnet.layer3.0.depthconv2
{ "name":"conv_24", "inc": 24 , "outc":24 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_23_out", "out":"conv_24_out", "relu":1 ,"in_dim":48 , "pt":"fe"}, # fnet.layer3.0.pointconv2
{ "name":"conv_25", "inc": 24 , "outc":96 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_24_out", "out":"conv_25_out", "relu":1 ,"in_dim":48 , "pt":"fe"}, # fnet.layer3.0.conv3
{ "name":"conv_26", "inc": 64 , "outc":96 , "ks":1  , "pad":0  , "sde":2  , "dw":0, "inp":"conv_16_out", "out":"conv_26_out", "relu":0 ,"in_dim":96 , "pt":"fe"}, # fnet.layer3.0.conv4
{ "name":"res_add", "inc": 32 , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_25_out", "out":"conv_26_out", "relu":1 ,"in_dim":48 , "pt":"fe", "cfg":"conv_26"}, # res_add3

{ "name":"conv_32", "inc": 96 , "outc":96 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_30_out", "out":"conv_32_out", "relu":0 ,"in_dim":48 , "pt":"fe"}, # fnet.conv2

{ "name":"matmul_1", "a": "conv_32_out" , "b":"conv_32_out", "d":"NULL", "c":"corr_1" , "aT":0,  "bT":1, "I":2304, "J":2304, "K":96, "aScale":"1.0", "bScale":"1.0", "biasScale":"1.0", "out_dim":48, "pt":"corrgen" }, # corr first layer
{ "name":"pool_1 ", "inc": 2304,"outc":2304,"ks":2  , "pad":0  , "sde":2  , "dw":1, "inp":"corr_1 ", "out":"corr_2 ", "relu":0, "in_dim":48, "pt":"corrgen" }, # corr layer 2
{ "name":"pool_2 ", "inc": 2304,"outc":2304,"ks":2  , "pad":0  , "sde":2  , "dw":1, "inp":"corr_2 ", "out":"corr_3 ", "relu":0, "in_dim":24, "pt":"corrgen" }, # corr layer 3
{ "name":"pool_3 ", "inc": 2304,"outc":2304,"ks":2  , "pad":0  , "sde":2  , "dw":1, "inp":"corr_3 ", "out":"corr_4 ", "relu":0, "in_dim":12, "pt":"corrgen" }, # corr layer 4

{ "name":"code1  "},

#inp=net+2304*96, out=net+2304*160, flow=net+2304*240, q_pre=net+2304*242
{ "name":"conv_66", "inc": 196, "outc":96 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"corr        ", "out":"cor           ", "relu":0, "in_dim":48, "pt":"update"}, # ud.encoder.convc1
{ "name":"conv_67", "inc": 2  , "outc":64 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"net+2304*240", "out":"flo1          ", "relu":0, "in_dim":48, "pt":"update"}, # ud.encoder.convf1
{ "name":"conv_68", "inc": 64 , "outc":32 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"flo1        ", "out":"cor+2304*96   ", "relu":0, "in_dim":48, "pt":"update"}, # ud.encoder.convf2
{ "name":"conv_69", "inc": 128, "outc":128, "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"cor         ", "out":"conv_69_out   ", "relu":0, "in_dim":48, "pt":"update"}, # ud.encoder.depthconv
{ "name":"conv_70", "inc": 128, "outc":80 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"conv_69_out ", "out":"net+2304*140  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.encoder.pointconv
{ "name":"conv_71", "inc": 242, "outc":242, "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.depthconvz
{ "name":"conv_72", "inc": 242, "outc":96 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"update_dwout", "out":"net+2304*242  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.pointconvz
{ "name":"conv_73", "inc": 242, "outc":242 ,"ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.depthconvr
{ "name":"conv_74", "inc": 242, "outc":96 , "ks":3  , "pad":0  , "sde":1  , "dw":0, "inp":"update_dwout", "out":"z             ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.pointconvr
{ "name":"conv_75", "inc": 242, "outc":242, "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"net+2304*96 ", "out":"update_dwout  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.depthconvq
{ "name":"conv_76", "inc": 242, "outc":96 , "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"update_dwout", "out":"net           ", "relu":0, "in_dim":48, "pt":"update"}, # ud.gru.pointconvq
{ "name":"conv_77", "inc": 96 , "outc":96 , "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"net         ", "out":"update_dwout  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.flow_head.depthconv1
{ "name":"conv_78", "inc": 96 , "outc":128, "ks":1  , "pad":0  , "sde":1  , "dw":0, "inp":"update_dwout", "out":"f_head        ", "relu":0, "in_dim":48, "pt":"update"}, # ud.flow_head.pointconv1
{ "name":"conv_79", "inc": 128, "outc":128, "ks":3  , "pad":1  , "sde":1  , "dw":1, "inp":"f_head      ", "out":"update_dwout  ", "relu":1, "in_dim":48, "pt":"update"}, # ud.flow_head.depthconv2
{ "name":"conv_80", "inc": 128, "outc":2  , "ks":3  , "pad":0  , "sde":1  , "dw":0, "inp":"update_dwout", "out":"net+2304*240  ", "relu":0, "in_dim":48, "pt":"update"}, # ud.flow_head.pointconv2

]


def initial_config(model):
    index = 0
    for name,module in model.named_modules():
        params = {}
        if 'quant' not in name and 'conv' in name:
            print('{{ \"name\":\"conv_{:<2}\", \"inc\": {:<3}, \"outc\":{:<3}, \"ks\":{:<3}, \"pad\":{:<3}, \"sde\":{:<3}, \"dw\":{:<1}, \"inp\":\"conv_{:<2}_out\", \"out\":\"conv_{:<2}_out\", \"relu\":{} }}, # {}'.format(index, 
                                                                                module.in_channels, 
                                                                                module.out_channels, 
                                                                                module.kernel_size[0], 
                                                                                module.padding[0], 
                                                                                module.stride[0],
                                                                                module.groups == module.in_channels,
                                                                                index-1,
                                                                                index,
                                                                                0,
                                                                                name,))
            # print("\"name\":\"conv_{}\", \"input\":\"conv_{}\", \"output\":\"conv_{}\" \t,#{} ".format(index, index-1, index, name))
            index += 1

def matmulText(params: dict) -> str:
    aT = "true" if params["aT"] else "false"
    aStride = params["I"] if params["aT"] else params["K"]
    bT = "true" if params["bT"] else "false"
    bStride = params["K"] if params["bT"] else params["J"]
    return matmul_template.format(I=params["I"], J=params["J"], K=params["K"],
                                aArray=params["a"], bArray=params["b"], dArray=params["d"], cArray=params["c"],
                                aStride=aStride, bStride=bStride, 
                                aScale=params["aScale"], bScale=params["bScale"], biasScale=params["biasScale"],
                                aTrans=aT, bTrans=bT, pt=params["pt"], name=params["name"])



def convText(params: dict) -> str:
    if params["name"] == "res_add":
        return res_add.format(name=params["cfg"].strip(), output=params["out"].strip(), input=params["inp"].strip(), relu=params["relu"], pt=params["pt"])
    elif params["dw"]:
        return dw_conv.format(name=params["name"].strip(), output=params["out"].strip(), input=params["inp"].strip(), relu=params["relu"], pt=params["pt"])
    elif params["ks"] == 1:
        return pw_conv.format(name=params["name"].strip(), output=params["out"].strip(), input=params["inp"].strip(), relu=params["relu"], pt=params["pt"])
    else:
        return regular_conv.format(name=params["name"].strip(), output=params["out"].strip(), input=params["inp"].strip(), relu=params["relu"], pt=params["pt"])


def convConfigText(params: dict) -> str:
    inPicSize = params["batch"]*params["in_dim"]*params["in_dim"]
    outPicSize = params["batch"]*params["out_dim"]*params["out_dim"]
    
    if "p_size" in params:
        pool_size, pool_stride, pool_padding = params["p_size"], params["p_sde"], params["p_pad"]
    else:
        pool_size, pool_stride, pool_padding = 1, 1, 0;

    configText = conv_config.format(
                            name=params["name"].strip(),
                            batch_size=params["batch"],
                            in_dim=params["in_dim"], out_dim=params["out_dim"],
                            kernel_size=params["ks"],
                            in_channels=params["inc"],
                            out_channels=params["outc"],
                            stride=params["sde"],
                            padding=params["pad"],
                            bias=1, depthwise=params["dw"],
                            n_patches=params["in_dim"], patch_size=params["in_dim"]*params["out_dim"]*params["batch"],
                            output_scale=1, res_scale=1,
                            pool_size=pool_size, pool_stride=pool_stride, pool_padding=pool_padding, out_dim_pooled=params["out_dim"],
                            I=inPicSize, J=params["outc"], K=params["inc"]
                            )

    if params["dw"]:
        return configText + dw_conv_array.format(c=params["inc"], name=params["name"].strip()) + "\n"
    elif params["ks"] == 1:
        return configText + pw_conv_array.format(ic=params["inc"], oc=params["outc"], name=params["name"].strip()) + "\n"
    else:
        return configText + regular_conv_array.format(iksize=params["inc"]*params["ks"]*params["ks"], oc=params["outc"], name=params["name"].strip()) + "\n"
        

def updateMemory(memoryLocation: dict, config: dict, memoryName:list) -> dict:
    relatedParam = {
        "inp": ["in_dim", "inc"],
        "out": ["out_dim", "outc"]
    }
    for name in memoryName:
        mName = config[name].split("+")
        if len(mName) > 1:
            e_size, e_ch = mName[1].split("*")
            if int(e_size) != config["batch"] * config["in_dim"]**2:
                raise Exception("value error")
        else:
            e_ch = 0
        strName = mName[0].strip()
        m_size, m_ch = memoryLocation.get(strName, (0, 0))
        f_size, f_ch = config["batch"] * config[relatedParam[name][0]]**2, config[relatedParam[name][1]] + int(e_ch)
        memoryLocation[strName] = (f_size, f_ch) if f_size*f_ch > m_size*m_ch else (m_size, m_ch)
    return memoryLocation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    model = RAFT(args)

    initial_config(model)

    batch = 1
    memoryLocation = {}

    with open("./gemmini/raft.template.c", "r") as f:
        ctemplateText = f.read()
    ctemplateHead, ctemplateTail = ctemplateText.split("//generate code here\n", 2)
    with open("./gemmini/raft.template.h", "r") as f:
        htemplateText = f.read()
    htemplateHead, htemplateTail = htemplateText.split("//generate code here\n", 2)


    with open("./gemmini/raft.c", "w") as f:
        f.write(ctemplateHead)
        for config in layer_config:
            config["batch"] = batch

            if "matmul" in config["name"]:
                f.write(matmulText(config))
            elif "conv" in config["name"] or "res_add" in config["name"] or "pool" in config["name"]:
                config["out_dim"] = config["in_dim"] // config["sde"]

                memoryLocation = updateMemory(memoryLocation, config, ["inp", "out"])
                
                f.write(convText(config))
            else:
                f.write(usercode[config["name"].strip()])

        f.write(ctemplateTail)

    with open("./gemmini/raft_params.h", "w") as f:
        f.write(htemplateHead)
        for config in layer_config:
            if "conv" in config["name"] or "pool" in config["name"]:
                f.write(convConfigText(config))

        for memory, volume in memoryLocation.items():
            f.write(memory_array.format(memory=memory.strip(), size=volume[0], channel=volume[1]))
        f.write(htemplateTail)

