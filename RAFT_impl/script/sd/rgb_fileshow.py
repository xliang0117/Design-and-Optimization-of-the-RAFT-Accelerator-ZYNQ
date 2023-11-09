import numpy as np
import cv2
import sys

for filename in ["image1.bin", "image2.bin"]:
    image = np.fromfile(filename, dtype=np.uint8)
    image = image.reshape(384,384,3)
    image = np.pad(image,((0,0), (0,0), (0,1)))
    image.tofile("ex_"+filename)

flowRGBA = np.fromfile("flow_RGBA.bin", dtype=np.uint8)
flowRGBA = flowRGBA.reshape((1080,1920,4))
print(flowRGBA.shape)
image = flowRGBA[:,:,0:3]
print(image[192,192,:])
# image = np.ones(flowRGBA.shape) * 255
cv2.imshow('image', image[:, :, [1,2,0]]/255.0)
if cv2.waitKey() == 27:
    sys.exit()

