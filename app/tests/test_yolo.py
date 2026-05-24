import cv2

from ultralytics import YOLO

from insightface.app import FaceAnalysis


class AnalysisFaceManager:
    def __init__(
        self, model_name: str = 'buffalo_s', providers: list = []
    ) -> None:
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.faces_vetors: dict = {}

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


class YOLOManager:
    def __init__(
        self, yolo_name: str = 'yolov8n.pt', track_name: str = 'bytetrack.yaml'
    ) -> None:
        self.model = YOLO(yolo_name, task='detect')
        self.algorithm_track = track_name

    def detect(self, frame) -> list:
        return self.model.predict(frame)

    def _track_frame(self, frame):
        return self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=self.algorithm_track,
            device='cpu',
            iou=0.5,
            imgsz=640,
            conf=0.4,
            verbose=False,
        )


class AppVision:
    def __init__(self) -> None:
        self.tracker = YOLOManager(yolo_name='yolov8n_openvino_model/')
        self.persons = {}

    def run(self, img_path: str) -> None:
        _img = cv2.imread(img_path)
        if _img is None:
            return

        frame = cv2.resize(_img, (960, 640))

        results = self.tracker._track_frame(frame)

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy
            ids = result.boxes.id

            for box, track_id in zip(boxes, ids):
                pessoa_recortada = self.cut_frame(frame, box)

                window_name = f'Pessoa ID {track_id}'
                # cv2.imshow(window_name, pessoa_recortada)

        print('Pressione qualquer tecla em uma janela de imagem para fechar.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cut_frame(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = [max(0, x1), max(0, y1)]

        crop = frame[y1:y2, x1:x2]
        return crop

    def _get_atributes(self, result):
        return {
            'boxes': result.boxes.xyxy.cpu().numpy(),
            'confidence': result.boxes.conf.cpu().numpy(),
        }

    def _get_ids(self, result, boxes):
        if result.boxes.id is not None:
            return result.boxes.id.cpu().numpy().astype(int)
        else:
            return [None] * len(boxes)


if __name__ == '__main__':
    app = AppVision()

    app.run('app/assets/modelo.png')

    print(f'valores: {app.persons}')
