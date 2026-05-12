from fer.fer import FER
import cv2
import numpy as np

# Inicializa o detector
detector = FER(mtcnn=False)

# Carrega imagem
img = cv2.imread('app/assets/rian.jpeg')

# Garante que a imagem foi carregada e converte explicitamente
if img is None:
    print('Erro: imagem não encontrada')
else:
    img = np.array(
        img
    )  # conversão explícita para ndarray, resolve o type error

    resultado = detector.detect_emotions(img)
    print(resultado)
