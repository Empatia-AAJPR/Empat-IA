import cv2
from insightface.app import FaceAnalysis


class AnalysisFaceManager:
    def __init__(
        self, model_name: str = 'buffalo_s', providers: list = []
    ) -> None:
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.faces_vetors: dict = {}

    def parser_images(self, images: list):
        for img in images:
            _img = cv2.imread(img)

            if _img is None:
                print(f'image {img} is None')
                continue

            faces = self.app.get(_img)

            if len(faces) > 0:
                embedding = faces[0].normed_embedding
                print(f'Embedding: {embedding}')

                self.faces_vetors[img.split('/')[-1]] = embedding

            else:
                print('nenhum rosto detectado')
                continue


if __name__ == '__main__':
    face_analyser = AnalysisFaceManager(providers=['CPUExecutionProvider'])
    face_analyser.parser_images(
        [
            'app/assets/antonio.jpeg',
            'app/assets/arthur.jpeg',
            'app/assets/rian.jpeg',
            'app/assets/pablo.jpeg',
            'app/assets/joao.jpeg',
        ]
    )

    print(f'Vetores faciais: {face_analyser.faces_vetors}')
