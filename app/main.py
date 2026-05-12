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
    def __init__(self, url_capture: str | int) -> None:
        self.cap = cv2.VideoCapture(url_capture)
        self.tracker = YOLOManager(yolo_name='yolov8n_openvino_model/')
        self.persons = {}

    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 640))

            results = self.tracker._track_frame(frame)

            for result in results:
                attr = self._get_atributes(result)

                boxes = attr['boxes']
                confs = attr['confidence']
                ids = self._get_ids(result, boxes)

                for box, conf, id in zip(boxes, confs, ids):
                    if id is not None:
                        id = int(id)

                        self.persons[id] = {
                            'box': list(
                                map(lambda x: round(float(x), 2), box)
                            ),
                            'conf': round(float(conf), 2),
                        }

                    print(f'{self.persons}\n')

                    cv2.imshow('EmpatIA', result.plot())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

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
    app = AppVision(url_capture='http://192.168.1.161:8080/video')

    app.run()