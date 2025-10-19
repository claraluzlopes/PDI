import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Ler a imagem 
img = cv2.imread("limiarizacao/folha.jpg", 0)  

# 2. Pré-processamento: suavizar ruído e riscos finos
img_blur = cv2.medianBlur(img, 5)  # desfoque mediano para preservar bolinhas
cv2.imwrite("limiarizacao/folha_blur.jpg", img_blur)

# 3. Equalizar histograma para realçar manchas
img_eq = cv2.equalizeHist(img_blur)
cv2.imwrite("limiarizacao/folha_equalizada.jpg", img_eq)


# Global threshold
_, th_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("limiarizacao/saida_threshold_global.jpg", th_global)

# Otsu threshold
_, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("limiarizacao/saida_threshold_otsu.jpg", th_otsu)

# Adaptive threshold
th_adapt = cv2.adaptiveThreshold(
    img_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
cv2.imwrite("limiarizacao/saida_threshold_adaptativa.jpg", th_adapt)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
th_clean = cv2.morphologyEx(th_adapt, cv2.MORPH_OPEN, kernel)
cv2.imwrite("limiarizacao/folha_clean.jpg", th_clean)

# 3. Detectar contornos usando a imagem binária do Otsu 
contours, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_filtrados = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 20 < area < 200:  # ajustar limites conforme tamanho das bolinhas
        contours_filtrados.append(cnt)

# Copiar a imagem original para desenhar os contornos
img_contornos = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contornos, contours_filtrados, -1, (0, 0, 255), 2)
cv2.imwrite("limiarizacao/saida_contornos.jpg", img_contornos)

print(f"Número de manchas detectadas: {len(contours_filtrados)}")

#Mostrar o histograma de intensidades
plt.figure(figsize=(10,4))
plt.hist(img.ravel(), bins=256, range=[0,256])
plt.title('Histograma de Intensidades')
plt.xlabel('Nível de Cinza')
plt.ylabel('Frequência')
plt.show(block=False)   # não bloqueia
plt.pause(0.001)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(th_clean, cmap='gray')
plt.title("Threshold adaptativo + limpeza")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title(f"Manchas detectadas: {len(contours_filtrados)}")
plt.axis('off')

plt.tight_layout()
plt.show()

"""
Se várias folhas de uma mesma planta apresentam porcentagens próximas de área afetada, 
pode-se estimar que a planta como um todo esteja em torno desse nível de contaminação. 
Em escala maior, ao selecionar folhas de diferentes plantas de um cultivo, 
a média da área contaminada serve como indicador do estado fitossanitário do cultivo inteiro
"""