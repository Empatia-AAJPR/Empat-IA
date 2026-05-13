import cv2

from insightface.app import FaceAnalysis
import numpy as np


class AnalysisFaceManager:
    def __init__(
        self, model_name: str = 'buffalo_s', providers: list = [], conf: float = 0.5
    ) -> None:
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(320, 320))
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
    
    def get_face_person(self, base, crop_frame):
        img_base = cv2.imread(base)

        img_detect = crop_frame

        if img_base is None:
            return
        
        face_base = self.app.get(img_base)

        face_detect = self.app.get(img_detect)

        embedding_base = face_base[0].normed_embedding
        embedding_detect = face_detect[0].normed_embedding

        similarity = np.dot(embedding_base, embedding_detect)

        return self.is_same_person(similarity)
    
    def is_same_person(self, result):
        if result >= self.conf:
            return True
        
        return False
    