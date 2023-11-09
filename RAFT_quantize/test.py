from operator import mod
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from utils.utils import InputPadder



DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float() # get the tensor 
    return img[None].to(DEVICE)


if __name__ == '__main__':
    path = "./demo-frames"
    images = glob.glob(os.path.join(path, '*.png')) + \
                glob.glob(os.path.join(path, '*.jpg'))
    print(images)
    images = sorted(images)
    image1 = load_image(images[0])
    print(images[0])
    print(image1)
    
