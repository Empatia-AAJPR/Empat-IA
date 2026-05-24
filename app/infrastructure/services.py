from domain.servicies import ICompaterService
import cv2
import base64
import numpy as np


class CompacterService(ICompaterService):
    """
    Serviço de Infraestrutura - CompacterService.

    Classe utilitária responsável por converter imagens abertas pelo OpenCV
    em texto (Base64) e vice-versa. Essa conversão é necessária porque o
    banco de dados Redis só armazena texto/strings, enquanto os modelos de
    Inteligência Artificial (YOLO, InsightFace) só entendem matrizes do NumPy.
    - encoded_img: Pega uma imagem da câmera, compacta em formato JPEG (para
                   diminuir o tamanho) e transforma em uma string Base64 pronta
                   para ser salva no banco.
    - decoded_img: Pega o texto em Base64 vindo do Redis, transforma de volta
                   em bytes, recria a matriz binária (NumPy) e reconstrói a
                   imagem colorida original para ser processada pela IA.
    """

    @staticmethod
    def encoded_img(img):
        if img is None:
            return
        _, img_enconded = cv2.imencode('.jpeg', img)
        return base64.b64encode(img_enconded)

    @staticmethod
    def decoded_img(img_base64):
        if img_base64 is None:
            return
        img_bytes = base64.b64decode(img_base64)
        np_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
