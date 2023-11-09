import sys
sys.path.append("./script")

from code_gen import hlsGenerator
from file_gen import fileWrapper, fileTemplate
from regularSeqConfig import ddrConfig, layerConfig
import argparse

import colored_traceback
colored_traceback.add_hook()



def demo(args):
    hlsGen = hlsGenerator(args)
    layerConfig[0]["sheduled"] = True
    if layerConfig[22].get("code", None):
        layerConfig[22]["code"] = "C"
    else:
        raise Exception("key code not existed")
    fileWrapper(inText=hlsGen.getTextFromConfig(layerConfig, ddrConfig), fileTemplate=fileTemplate)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inParallel', help="input channel parallelism", default=32)
    parser.add_argument('--outParallel', help="output channel parallelism", default=32)
    parser.add_argument('--tileDims', help='on-chip buffer dims', default=48)
    args = parser.parse_args()

    demo(args)