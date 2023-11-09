import onnx
from onnx import numpy_helper
import numpy as np
import math
import colored_traceback
colored_traceback.add_hook()

if __name__ == "__main__":
    model = onnx.load('onnx_model.onnx')
    onnx.checker.check_model(model)
    initializer  = model.graph.initializer
    onnx_params = {}
    for w in initializer:
        w_np = numpy_helper.to_array(w)
        onnx_params[w.name] = w_np
        print(w.name)

    nodes = model.graph.node
    nodeDict = {}
    outputSource = {}
    convLayer = {}
    convLayerParam = {}
    for n in nodes:
        nodeDict[n.name] = {"name":n.name, "input":n.input, "output":n.output}
        outputSource[n.output[0]] = n.name
        # if 'Quant' in n.name:
        #     print(type(n))
        #     print(n.name)
        #     print(n.input)
        #     print(n.attribute)
    for key, value in nodeDict.items():
        if "Conv" in key:
            print(nodeDict[outputSource[value["input"][2]]]["input"][0])
            input_quant = False if "Conv" in nodeDict[outputSource[value["input"][0]]]["name"] else True
            convLayer[nodeDict[outputSource[value["input"][2]]]["input"][0]] = {"input_quant":input_quant,
                                                                                "input":nodeDict[outputSource[value["input"][0]]]["input"],
                                                                                "weight":nodeDict[outputSource[value["input"][1]]]["input"],
                                                                                "bias":nodeDict[outputSource[value["input"][2]]]["input"]}
    print(convLayer)

    for key, value in convLayer.items():
        print(value["input"][1])
        if value["input_quant"]:
            input_scale = onnx_params[value["input"][1]]
            input_zeropoint = onnx_params[value["input"][2]]
        else:
            input_scale = 0
            input_zeropoint = 0
        convLayerParam[key] = {
            "input_scale":input_scale,
            "input_zeropoint":input_zeropoint,
            "weight":onnx_params[value["weight"][0]],
            "weight_scale":onnx_params[value["weight"][1]],
            "weight_zeropoint":onnx_params[value["weight"][2]],
            "bias":onnx_params[value["bias"][0]],
            "bias_scale":onnx_params[value["bias"][1]],
            "bias_zeropoint":onnx_params[value["bias"][2]],
        }

    for key, value in convLayerParam.items():
        print(key)
        print(math.log2(value["weight_scale"]))
        print(math.log2(value["bias_scale"]))