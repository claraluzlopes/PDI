"""
EXERCICIO 02 - Alterações de canais: remoção, troca e reforço
"""

# bibliotecas necessarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "sample_data/joker.jpeg" #pega a imagem
img_bgr = cv2.imread(img_path) #le a imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #converte bgr em rgb

img_no_blue = img_rgb.copy()
img_no_blue[:, :, 2] = 0

img_swap_rb = img_rgb.copy()
img_swap_rb[:, :, [0, 2]] = img_swap_rb[:, :, [2, 0]]

img_boost_green = img_rgb.copy()
img_boost_green[:, :, 1] = np.clip(img_boost_green[:, :, 1] * 1.3, 0, 255).astype(np.uint8)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)  # 1 linha, 3 colunas, posição 1
plt.imshow(img_no_blue)
plt.title("Remover Canal Azul")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_swap_rb)
plt.title("Troca R ↔ B")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_boost_green)
plt.title("Verde +30%")
plt.axis("off")

plt.show()

cv2.imwrite("output02/no_blue.jpg", cv2.cvtColor(img_no_blue, cv2.COLOR_RGB2BGR))
cv2.imwrite("output02/swap_rb.jpg", cv2.cvtColor(img_swap_rb, cv2.COLOR_RGB2BGR))
cv2.imwrite("output02/boost_green.jpg", cv2.cvtColor(img_boost_green, cv2.COLOR_RGB2BGR))
