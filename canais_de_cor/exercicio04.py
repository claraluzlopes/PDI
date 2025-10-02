"""
EXERCICIO 04 - Cinza (média vs. perceptual) + equalização de histograma
"""

# bibliotecas necessarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "sample_data/lgbt.png" #pega a imagem
img_bgr = cv2.imread(img_path) #le a imagem
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #converte bgr em rgb

R = img_rgb[:, :, 0].astype(float)
G = img_rgb[:, :, 1].astype(float)
B = img_rgb[:, :, 2].astype(float)

# quando implementamos cinza por média fazemos a média de RGB, portanto, as três cores contribuem igualmente para a luminância final.
# o olho humano é mais sensível ao verde, menos ao vermelho e ainda menos ao azul, por isso, na primeira imagem a listra verde ficou a mais escura, depois a vermelha e a azul a mais clara.
gray_mean = ((R + G + B) / 3).astype(np.uint8)

#No Cinza (Perceptual) o contraste fica mais próximo da realidade.
gray_perc = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

gray_eq = cv2.equalizeHist(gray_perc)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(gray_mean, cmap="gray")
plt.title("Cinza (Média)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray_perc, cmap="gray")
plt.title("Cinza (Perceptual)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gray_eq, cmap="gray")
plt.title("Equalização de Histograma")
plt.axis("off")

plt.show()

cv2.imwrite("output04/ex4_cinza_media.jpg", gray_mean)
cv2.imwrite("output04/ex4_cinza_perceptual.jpg", gray_perc)
cv2.imwrite("output04/ex4_equalizacao.jpg", gray_eq)