import os

from typing import Any

""" CONFIGURAÇÃO DO AMBIENTE E ACELERAÇÃO POR HARDWARE (GPU):
 Define o caminho para os binários do Toolkit do NVIDIA CUDA instalados no Windows """
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin'
if os.path.exists(cuda_path):
    # Adiciona o diretório da CUDA ao PATH de DLLs do Python para que o ONNX Runtime encontre a GPU
    os.add_dll_directory(cuda_path)
try:
    # Tenta carregar a biblioteca CuDNN específica do pacote nvidia via Python
    import nvidia.cudnn

    # Adiciona os binários do CuDNN ao diretório de busca de DLLs do sistema
    os.add_dll_directory(
        os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'bin')
    )
except:
    # Caso ocorra erro na importação do pacote, ignora e prossegue (caso use a CPU)
    pass

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
        """
        Método Construtor: Configura os provedores de execução (CPU ou GPU)
        e inicializa o modelo 'buffalo_s' do InsightFace.
        """
        if model_root is None:
            model_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

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
        self.registers_model: dict = {}

    def register_faces(self, images: list[dict[str, Any]]):
        """
        Cadastro de Faces (Dicionário): Recebe um dicionário estruturado com chaves de nomes
        e valores de caminhos de imagens. Processa as fotos em disco e gera o banco de dados vetorial.
        """
        for img in images:
            _img = img.get('vector')

            if _img is None:
                continue

            faces = self.app.get(_img)

            if len(faces) > 0:
                embedding = faces[0].normed_embedding

                student_id = img.get('id')
                if not student_id:
                    continue

                self.registers_model[str(student_id)] = {
                    'name': img.get('name', 'Sem Nome'),
                    'embedding': embedding,
                }

    def get_face_person(self, crop_frame):
        """
        Reconhecimento Facial em Tempo Real: Recebe o recorte (crop) da face obtido na câmera,
        extrai o seu vetor numérico e faz o produto escalar contra todos os rostos conhecidos.
        """
        img_detect = crop_frame
        face_detect = self.app.get(img_detect)
        if not face_detect or len(face_detect) == 0:
            return None
        embedding_detect = face_detect[0].normed_embedding
        for k, value in self.registers_model.items():
            similarity = np.dot(value['embedding'], embedding_detect)
            print(f'SIMILARIDADE: {similarity}')
            if similarity >= 0.5:
                return k

    def is_same_person(self, result):
        """
        Validador de Confiança: Método utilitário para validar se um resultado numérico
        de similaridade atinge a confiança global estipulada em 'self.conf'.
        """
        if result >= self.conf:
            return True

        return False
