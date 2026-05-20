from abc import ABC, abstractmethod


class ICompaterService(ABC):
    """
    Interface de Domínio - ICompaterService.
    Contrato abstrato que define as operações necessárias para a conversão e 
    compactação de imagens no sistema. Garante o desacoplamento entre as regras 
    de negócio e as bibliotecas de manipulação de imagem (como OpenCV e Base64).
    - encoded_img: Deve receber uma imagem (matriz/array) e convertê-la em uma 
                   string binária/texto codificado para armazenamento seguro no banco.
    - decoded_img: Deve receber a string codificada (Base64) vinda do banco e 
                   reconstruir a imagem original em formato de matriz para processamento da IA.
    """
    @staticmethod
    @abstractmethod
    def encoded_img(img):
        ...

    @staticmethod
    @abstractmethod
    def decoded_img(img_base64):
        ...
