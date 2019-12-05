from imaging_system_responsefunction import get_response_function_and_log_irradiance, w
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from process_images import hdr_image_generator

image_paths = sorted(glob.glob("photos/*.jpg"))

TARGET_HEIGHT = 364
TARGET_WIDTH = 648

print(image_paths)

images = np.array([
    cv2.cvtColor(cv2.resize(cv2.imread(path), (TARGET_WIDTH, TARGET_HEIGHT)),
                 cv2.COLOR_BGR2GRAY).astype('float32') for path in image_paths
])

SAMPLE_SIZE = 2500
p_shutters = [1. / 60., 1. / 30., 1. / 15., 1. / 8.]
s_shutters = [1. / 160., 1. / 125., 1. / 80., 1. / 60., 1. / 40., 1. / 15.]
l_shutters = [1. / 250., 1. / 125., 1. / 60., 1. / 30., 1. / 15., 1. / 8.]
"""
    Creating output images.
"""

HDR = hdr_image_generator(images, p_shutters, lamb=3200.)
cv2.imwrite('output/hdr.png', HDR)

print('Gray HDR created...')

images = np.array([
    cv2.resize(cv2.imread(path),
               (TARGET_WIDTH, TARGET_HEIGHT)).astype('float32')
    for path in image_paths
])

HDR = hdr_image_generator(images, p_shutters, channels=3, lamb=9200.)
cv2.imwrite('output/hdr_color.png', HDR)

print('RGB HDR created...')