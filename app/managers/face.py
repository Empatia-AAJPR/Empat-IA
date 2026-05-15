import cv2

from insightface.app import FaceAnalysis
import numpy as np

import os


class AnalysisFaceManager:
    def __init__(
        self,
        use_gpu: bool,
        model_root: str = 'models/',
        providers: list = [],
        conf: float = 0.5,
    ) -> None:
        if model_root is None:
            model_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.app = FaceAnalysis(
            name='buffalo_s', 
            root=model_root,
            providers=providers,
            allowed_modules=['detection', 'recognition'],
        )
        self.app.prepare(ctx_id=-1 if use_gpu is False else 0, det_size=(320, 320))
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
