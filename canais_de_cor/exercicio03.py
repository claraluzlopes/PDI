"""
EXERCICIO 03 - Transformações geométricas: translação, rotação e escala
"""

# bibliotecas necessarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "sample_data/espias.jpg" #pega a imagem
img_bgr = cv2.imread(img_path) #le a imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #converte bgr em rgb

h, w, = img_rgb.shape[:2]

M_trans = np.float32([[1, 0, 40], [0, 1, 25]])
img_trans = cv2.warpAffine(img_rgb, M_trans, (w, h))

center = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, 20, 1)
img_rot = cv2.warpAffine(img_rgb, M_rot, (w, h))

img_scale = cv2.resize(img_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_trans)
plt.title("Translação (+40, +25)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_rot)
plt.title("Rotação (+20°)")
plt.axis("off")

plt.figure(figsize=(12, 12))
plt.imshow(img_scale)
plt.title("Escala 1.5x")
plt.axis("off")

plt.show()

cv2.imwrite("output03/saida_translacao.jpg", cv2.cvtColor(img_trans, cv2.COLOR_RGB2BGR))
cv2.imwrite("output03/saida_rotacao.jpg", cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR))
cv2.imwrite("output03/saida_escala.jpg", cv2.cvtColor(img_scale, cv2.COLOR_RGB2BGR))

