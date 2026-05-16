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
