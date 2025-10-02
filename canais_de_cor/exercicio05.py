"""
EXERCICIO 05 - Cisalhamento (shear) e recorte (crop)
"""

# bibliotecas necessarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "sample_data/gagaeburton.webp" #pega a imagem
img_bgr = cv2.imread(img_path) #le a imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #converte bgr em rgb

h, w = img_rgb.shape[:2]

Sx = 0.3
M_shear = np.float32([[1, Sx, 0], [0, 1, 0]])
img_shear = cv2.warpAffine(img_rgb, M_shear, (w + int(Sx*h), h))  # ajustar largura

crop_h = int(0.6 * h)
crop_w = int(0.6 * w)
start_row = (h - crop_h) // 2
start_col = (w - crop_w) // 2
img_crop = img_rgb[start_row:start_row + crop_h, start_col:start_col + crop_w]

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_shear)
plt.title("Shear (Sx=0.3)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_crop)
plt.title("Crop Central (60%)")
plt.axis("off")

plt.show()

cv2.imwrite("output05/shear.jpg", cv2.cvtColor(img_shear, cv2.COLOR_RGB2BGR))
cv2.imwrite("output05/crop.jpg", cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
