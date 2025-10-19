import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import os

input_path = 'sample_data/Imagens.jpg'  
output_dir = 'output'                   

# carregar imagem
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {input_path}")

# escala de cinza 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# melhorar resolução
upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Equalização de histograma 
equalized = cv2.equalizeHist(upscaled)

# Expansão morfológica 
kernel = np.ones((3, 3), np.uint8)
expanded = cv2.dilate(equalized, kernel, iterations=1)

# Detecção de rostos 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(
    expanded,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30)
)

# desenhar retângulos
output = cv2.cvtColor(expanded, cv2.COLOR_GRAY2RGB)
for (x, y, w, h) in faces:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir
fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].imshow(gray, cmap='gray')
axs[0].set_title('Escala de Cinza')
axs[1].imshow(expanded, cmap='gray')
axs[1].set_title('Ampliada e Equalizada')
axs[2].imshow(output)
axs[2].set_title(f'Rostos Detectados ({len(faces)})')
for ax in axs:
    ax.axis('off')
plt.show()

cv2.imwrite(os.path.join(output_dir, '1_gray.jpg'), gray)
cv2.imwrite(os.path.join(output_dir, '2_upscaled_equalized.jpg'), equalized)
cv2.imwrite(os.path.join(output_dir, '3_expanded.jpg'), expanded)
cv2.imwrite(os.path.join(output_dir, '4_faces_detected.jpg'), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

