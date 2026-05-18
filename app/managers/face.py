import os
import sys


cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
try:
    import nvidia.cudnn
    os.add_dll_directory(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin"))
except:
    pass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

class AnalysisFaceManager:
    def __init__(
        self,
        use_gpu: bool,
        model_root: str = 'models/',
        providers: list = None,  
        conf: float = 0.5,
    ) -> None:
        if model_root is None:
            model_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name='buffalo_s', 
            root=model_root,
            providers=providers,  
            allowed_modules=['detection', 'recognition'],
        )
        
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(320, 320))
        self.faces_vetors: dict = {}
        self.conf = conf

    def parser_images(self, images: list) -> None:
        for img in images:
            _img = cv2.imread(img)

            if _img is None:
                print(f'image {img} is None')
                continue

            faces = self.app.get(_img)

            if len(faces) > 0:
                embedding = faces[0].normed_embedding

                self.faces_vetors[img.split('/')[-1]] = embedding

            else:
                print('nenhum rosto detectado')
                continue

    def register_faces(self, images: list[dict[str, any]]):
        
        picture = images[0]
        for nome, caminho in picture.items():
            img = cv2.imread(caminho)
            if img is None:
                print(f" Erro: Não foi possível ler a imagem de {nome} no caminho: {caminho}")
                continue
            face = self.app.get(img)
            if len(face) > 0:
                embedding_face = face[0].normed_embedding
                self.faces_vetors[nome] = embedding_face

        
    def get_face_person(self, crop_frame):
        
        img_detect = crop_frame
        face_detect = self.app.get(img_detect)
        if not face_detect or len(face_detect) == 0:
            return None
        
        embedding_detect = face_detect[0].normed_embedding
        for k ,value in self.faces_vetors.items():

            similarity = np.dot(value, embedding_detect)
            if similarity > 0.5:
                return k
            

    def is_same_person(self, result):
        if result >= self.conf:
            return True

        return False


if __name__ == '__main__':
    analisar = AnalysisFaceManager(True)

    analisar.register_faces([{ 'Antonio' : r'Empat-IA\app\assets\antonio.jpeg' ,
                                'Arthur' : r'Empat-IA\app\assets\arthur.jpeg',
                                'Joao' : r'Empat-IA\app\assets\joao.jpeg',
                                'Pablo' : r'Empat-IA\app\assets\pablo.jpeg',
                                'Rian' : r'Empat-IA\app\assets\rian.jpeg',}])
