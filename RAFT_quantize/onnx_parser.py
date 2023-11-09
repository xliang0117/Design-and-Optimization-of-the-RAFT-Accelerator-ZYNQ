import onnx
from onnx import numpy_helper
import argparse
import torch
import numpy as np
import math
import copy
from collections import OrderedDict
import colored_traceback
colored_traceback.add_hook()

layer_mapping = OrderedDict([
    ("fnet.depthconv1","EA3"),
    ("fnet.pointconv1","EA1"),
    ("fnet.layer2.0.conv1","EE"),
    ("fnet.layer2.0.depthconv2","EF3"),
    ("fnet.layer2.0.pointconv2","EF1"),
    ("fnet.layer2.0.conv3","EG"),
    ("fnet.layer2.0.conv4","EP1"),
    ("fnet.layer3.0.conv1","EH"),
    ("fnet.layer3.0.depthconv2","EI3"),
    ("fnet.layer3.0.pointconv2","EI1"),
    ("fnet.layer3.0.conv3","EJ"),
    ("fnet.layer3.0.conv4","EP2"),
    ("fnet.conv2","EK"),
    ("cnet.depthconv1","CA3"),
    ("cnet.pointconv1","CA1"),
    ("cnet.layer2.0.conv1","CE"),
    ("cnet.layer2.0.depthconv2","CF3"),
    ("cnet.layer2.0.pointconv2","CF1"),
    ("cnet.layer2.0.conv3","CG"),
    ("cnet.layer2.0.conv4","CP1"),
    ("cnet.layer3.0.conv1","CH"),
    ("cnet.layer3.0.depthconv2","CI3"),
    ("cnet.layer3.0.pointconv2","CI1"),
    ("cnet.layer3.0.conv3","CJ"),
    ("cnet.layer3.0.conv4","CP2"),
    ("cnet.conv2.net","CK1"),
    ("cnet.conv2.inp","CK2"),
    ("update_block.encoder.convc1","A"),
    ("update_block.encoder.convf1","B"),
    ("update_block.encoder.convf2","C"),
    ("update_block.encoder.depthconv","D3"),
    ("update_block.encoder.pointconv","D1"),
    ("update_block.gru.depthconvz","F3"),
    ("update_block.gru.pointconvz","F1"),
    ("update_block.gru.depthconvr","E3"),
    ("update_block.gru.pointconvr","E1"),
    ("update_block.gru.depthconvq","G3"),
    ("update_block.gru.pointconvq","G1"),
    ("update_block.flow_head.depthconv1","H3"),
    ("update_block.flow_head.pointconv1","H1"),
    ("update_block.flow_head.conv2","I"),
])

def exportStateDict(convParam:dict, test=False):
    state_dict = {}
    state_dict_original = {}
    for key, value in convParam.items():
        splitKey = key.split(".")
        weightName = key + ".weight"
        biasName = key + ".bias"
        scaleName = ".".join(splitKey[0:-1] + [splitKey[-1][0] + (splitKey[-1][-2] if splitKey[-1][-2] != "v" else "") + splitKey[-1][-1] + "act"])
        # print(scaleName)
        # print("log:{}, bias scale:{}".format(math.log2(value["bias_scale"]),value["bias_scale"]))
        weight = np.ones(value["weight"].shape)*value["weight_scale"] if test else value["weight"]
        bias = np.ones(value["bias"].shape)*value["input_scale"]*value["weight_scale"] if test else value["bias"]
        state_dict[weightName] = torch.tensor(weight/(value["weight_scale"]))
        state_dict[biasName] = torch.clamp(torch.floor(torch.tensor(bias/value["input_scale"]/value["weight_scale"] + 0.5)), min=-32768, max=32767)
        state_dict[scaleName+".input_scale"] = torch.tensor(value["input_scale"])
        state_dict[scaleName+".weight_scale"] = torch.tensor(value["weight_scale"])
        state_dict[scaleName+".output_scale"] = torch.tensor(value["output_scale"])
            
        state_dict_original[weightName] = torch.tensor(weight)
        state_dict_original[biasName] = torch.tensor(bias)
    torch.save(state_dict, "raft-impl.pth")
    torch.save(state_dict_original, "raft-impl-original.pth")

def exportWeight(convParam:dict, test=False):
    addr = 0
    # split cnet.conv2 to net and inp
    convParam["cnet.conv2.net"] = copy.deepcopy(convParam["cnet.conv2"])
    convParam["cnet.conv2.inp"] = copy.deepcopy(convParam["cnet.conv2"])
    convParam["cnet.conv2.net"]["weight"] = convParam["cnet.conv2.net"]["weight"][0:96,:,:,:]
    convParam["cnet.conv2.net"]["bias"] = convParam["cnet.conv2.net"]["bias"][0:96]
    convParam["cnet.conv2.inp"]["weight"] = convParam["cnet.conv2.inp"]["weight"][96:160,:,:,:]
    convParam["cnet.conv2.inp"]["bias"] = convParam["cnet.conv2.inp"]["bias"][96:160]

    corr_weight = convParam["update_block.encoder.convc1"]["weight"]
    corr_part = np.split(corr_weight, 4, axis=1)
    corr_weight = np.concatenate([np.transpose(x.reshape(96,7,7), (0,2,1)).reshape(96, 49, 1, 1) for x in corr_part], axis=1)
    corr_part = np.split(corr_weight, 4, axis=1)
    new_slice = []
    new_slice.append(np.concatenate([x[:,0:32,:,:] for x in corr_part], axis=1))
    new_slice.append(np.concatenate([x[:,32:48,:,:] for x in corr_part], axis=1))
    new_slice.append(np.stack([x[:,48,:,:] for x in corr_part], axis=1))
    convParam["update_block.encoder.convc1"]["weight"] = np.concatenate(new_slice, axis=1)


    qpre_weight = convParam["update_block.gru.depthconvq"]["weight"][0:96,:,:,:]
    inp_weight = convParam["update_block.gru.depthconvq"]["weight"][96:242,:,:,:]
    inp_weight = np.pad(inp_weight, ((0, 14), (0, 0), (0, 0), (0, 0)), 'constant')
    convParam["update_block.gru.depthconvq"]["weight"] = np.concatenate([inp_weight, qpre_weight], axis=0)

    qpre_bias = convParam["update_block.gru.depthconvq"]["bias"][0:96]
    inp_bias = convParam["update_block.gru.depthconvq"]["bias"][96:242]
    inp_bias = np.pad(inp_bias, (0, 14), 'constant')
    convParam["update_block.gru.depthconvq"]["bias"] = np.concatenate([inp_bias, qpre_bias], axis=0)

    qpre_weight = convParam["update_block.gru.pointconvq"]["weight"][:,0:96,:,:]
    inp_weight = convParam["update_block.gru.pointconvq"]["weight"][:,96:242,:,:]
    inp_weight = np.pad(inp_weight, ((0, 0), (0, 14), (0, 0), (0, 0)), 'constant')
    convParam["update_block.gru.pointconvq"]["weight"] = np.concatenate([inp_weight, qpre_weight], axis=1)

    with open("export/param.bin", "wb") as f:
        for key in layer_mapping.keys():
            value = convParam[key]
            if test:
                weight = np.ones(value["weight"].shape)
                bias = np.ones(value["bias"].shape)
            else:
                weight = value["weight"]/value["weight_scale"]
                bias = np.clip(np.floor(value["bias"]/value["input_scale"]/value["weight_scale"] + 0.5), a_min=-32768, a_max=32767)
                 
            weight, bias = np.squeeze(weight), np.squeeze(bias)
            weight, bias = weight.astype(np.int8), bias.astype(np.int16)
            print("weight shape:{}".format(weight.shape))
            print("bias shape:{}".format(bias.shape))
            
            if len(weight.shape) == 3: # depthwise
                wch = (weight.shape[0] + 32 - 1) // 32 * 32
                weight = np.pad(weight, ((0, wch-weight.shape[0]), (0, 0), (0, 0)), 'constant')
                print("{}-w3:{}".format(key, addr))
                addr += np.prod(weight.shape) // 32
                weight = np.transpose(weight, (1, 2, 0))
                weight_sp = np.split(weight, wch//32, axis=2)
                list(map(lambda x: f.write(x.tobytes()), weight_sp))
            else:
                woch, wich = [(weight.shape[x]+32-1)//32*32 for x in range(2)]
                weight = np.pad(weight, ((0, woch-weight.shape[0]), (0, wich-weight.shape[1])), 'constant')
                print("{}-w1:{}".format(key, addr))
                addr += np.prod(weight.shape) // 32
                # weight = np.transpose(weight, (1,0))
                weight_sp = np.split(weight, wich//32, axis=1)
                for part in weight_sp:
                    part_sp = np.split(part, woch//32, axis=0)
                    list(map(lambda x: f.write(x.tobytes()), part_sp))
            
            bias_ch = (bias.shape[0] + 32 - 1) // 32 * 32
            bias = np.pad(bias, ((0, bias_ch-bias.shape[0]),), 'constant')
            print("{}-bias:{}".format(key, addr))
            addr += np.prod(bias.shape) // 16
            f.write(bias.tobytes())

            print("{} scale:{}\n".format(layer_mapping[key], math.log2(value["bias_scale"]/value["output_scale"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="use ones weight and bias")
    args = parser.parse_args()

    model = onnx.load('onnx_model.onnx')
    onnx.checker.check_model(model)
    initializer  = model.graph.initializer
    onnx_params = {}
    for w in initializer:
        w_np = numpy_helper.to_array(w)
        onnx_params[w.name] = w_np
        # print(w.name)

    nodes = model.graph.node 
    nodeDict = {}
    outputSource = {}
    inputSource = {}
    convLayer = {}
    convParam = {}
    for n in nodes:
        nodeDict[n.name] = {"name":n.name, "input":n.input, "output":n.output}
        outputSource[n.output[0]] = {"name":n.name, "input":n.input, "output":n.output}
        inputSource[n.input[0]] = {"name":n.name, "input":n.input, "output":n.output}
        # if 'Quant' in n.name:
        #     print(type(n))
        #     print(n.name)
        #     print(n.input)
        #     print(n.attribute)
    for key, value in nodeDict.items():
        if "Conv" in key:
            inputName = value["input"][0]
            weightName = value["input"][1]
            biasName = value["input"][2]
            convName = (outputSource[biasName]["input"][0]).replace(".bias", "")
            outputName = value["output"][0]
            if "Relu" in outputName or "Sigmoid" in outputName or "Tanh" in outputName or "input" in outputName:
                outputName = inputSource[outputName]["output"][0]

            # print(convName)
            convLayer[convName] = { "input":outputSource[inputName]["input"],
                                    "weight":outputSource[weightName]["input"],
                                    "bias":outputSource[biasName]["input"],
                                    "output":inputSource[outputName]["input"]}

    for key, value in convLayer.items():
        # print("{}:{}\n".format(key, value))
        if "scale" in value["input"][1]:
            input_scale = onnx_params[value["input"][1]]
            input_zeropoint = onnx_params[value["input"][2]]
        else:
            input_scale = 1
            input_zeropoint = 1

        if "scale" in value["output"][1]:
            output_scale = onnx_params[value["output"][1]]
            output_zeropoint = onnx_params[value["output"][2]]
        else:
            input_scale = 1
            input_zeropoint = 1

        convParam[key] = {
            "input_scale":input_scale,
            "input_zeropoint":input_zeropoint,
            "weight":onnx_params[value["weight"][0]],
            "weight_scale":onnx_params[value["weight"][1]],
            "weight_zeropoint":onnx_params[value["weight"][2]],
            "bias":onnx_params[value["bias"][0]],
            "bias_scale":onnx_params[value["bias"][1]],
            "bias_zeropoint":onnx_params[value["bias"][2]],
            "output_scale":output_scale,
            "output_zeropoint":output_zeropoint
        }

    for key, value in convParam.items():

        print(key)
        print("i: {}".format(math.log2(value["input_scale"])))
        print("w: {}".format(math.log2(value["weight_scale"])))
        print("b: {}".format(math.log2(value["bias_scale"])))
        print("o: {}".format(math.log2(value["output_scale"])))

    exportStateDict(convParam=convParam, test=args.test)
    exportWeight(convParam=convParam, test=args.test)

    