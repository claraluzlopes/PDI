import cv2
import numpy as np

# 1. Ler a imagem 
img = cv2.imread("folha.jpg")  
if img is None:
    raise ValueError("Não foi possível carregar a imagem.")

#  2. tons de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("saida_cinza.jpg", gray)

# 3. thresholding

# Global threshold
_, th_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("saida_threshold_global.jpg", th_global)

# Otsu threshold
_, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("saida_threshold_otsu.jpg", th_otsu)

# Adaptive threshold
th_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10
)
cv2.imwrite("saida_threshold_adaptativa.jpg", th_adapt)

# 4. Detectar contornos usando a imagem binária do Otsu 
contours, _ = cv2.findContours(th_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copiar a imagem original para desenhar os contornos
img_contornos = img.copy()
cv2.drawContours(img_contornos, contours, -1, (0, 0, 255), 2)
cv2.imwrite("saida_contornos.jpg", img_contornos)

#  5. Número de manchas  detectadas
num_manchas = len(contours)
print(f"Número de manchas detectadas: {num_manchas}")
"""
Se várias folhas de uma mesma planta apresentam porcentagens próximas de área afetada, 
pode-se estimar que a planta como um todo esteja em torno desse nível de contaminação. 
Em escala maior, ao selecionar folhas de diferentes plantas de um cultivo, 
a média da área contaminada serve como indicador do estado fitossanitário do cultivo inteiro
"""