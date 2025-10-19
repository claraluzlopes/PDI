import cv2 as cv
import numpy as np
import os

IMAGE_DIR = 'morfologia/sample_data'
OUTPUT_DIR = 'morfologia/output'

AREA_MIN = 800  # Área mínima do contorno para ser considerado uma moeda
AREA_MAX = 15000 # Área máxima do contorno para ser considerado uma moeda

# Tamanho do elemento estruturante (deve ser ímpar)
KERNEL_SIZE = 5 # Um valor maior remove mais ruído/fecha buracos maiores.

def processar_imagens(image_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
            # Caminho completo para a imagem
            image_path = os.path.join(image_dir, filename)

            # 1. Leitura da imagem
            image = cv.imread(image_path)
            if image is None:
                print(f"Erro: Não foi possível ler a imagem {filename}")
                continue

            output_image = image.copy()
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, binary_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
            opening = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel, iterations=2)
            img_bin_limpa = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
            contours, hierarchy = cv.findContours(img_bin_limpa, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for i, cnt in enumerate(contours):
                area = cv.contourArea(cnt)
                # Filtra contornos que são muito pequenos ou muito grandes
                if AREA_MIN < area < AREA_MAX:
                    valid_contours.append(cnt)
                    
                    # Desenha a bounding box e o rótulo na imagem de saída
                    x, y, w, h = cv.boundingRect(cnt)
                    cv.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    label_text = f"#{len(valid_contours)}"
                    cv.putText(output_image, label_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            total_moedas = len(valid_contours)
            print(f"{filename} -> moedas: {total_moedas}")
            output_filename = f"{os.path.splitext(filename)[0]}_out.png"
            output_path = os.path.join(output_dir, output_filename)
            cv.imwrite(output_path, output_image)

if __name__ == '__main__':
    processar_imagens(IMAGE_DIR, OUTPUT_DIR)
    print("\nProcessamento concluído.")