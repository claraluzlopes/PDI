"""
EXERCICIO 01 - CANAIS RGB
"""

# bibliotecas necessarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "sample_data/gagainred.webp" #pega a imagem
img_bgr = cv2.imread(img_path) #le a imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #converte bgr em rgb

R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

plt.figure(figsize=(12, 4))


# Canal Vermelho
plt.subplot(1, 3, 1)
plt.imshow(R, cmap="gray")
plt.title("Canal Vermelho (R)")
plt.axis("off")

# Canal Verde
plt.subplot(1, 3, 2)
plt.imshow(G, cmap="gray")
plt.title("Canal Verde (G)")
plt.axis("off")

# Canal Azul
plt.subplot(1, 3, 3)
plt.imshow(B, cmap="gray")
plt.title("Canal Azul (B)")
plt.axis("off")

plt.show()

cv2.imwrite("output01/saida_R.jpg", R)
cv2.imwrite("output01/saida_G.jpg", G)
cv2.imwrite("output01/saida_B.jpg", B)
