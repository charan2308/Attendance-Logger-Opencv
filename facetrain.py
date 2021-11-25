import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'Faces/train')

x_train = []
y_train = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jfif'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()
            print(label, path)
            # x_train.append(path)
            # y_train.append(label)
            pil_image = Image.open(path).convert('L')#convert to grayscale
            image_array=np.array(pil_image,'uint8')
            print(image_array)