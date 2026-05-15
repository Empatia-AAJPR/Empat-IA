import time

import cv2

from managers.emotions import EmotionDetecManager
from managers.face import AnalysisFaceManager
from managers.isolation_detector import DBScanManager
from managers.person_model import YOLOManager


class AppVision:
    def __init__(self, url_capture: str | int) -> None:
        self.cap = cv2.VideoCapture(url_capture)
        self.tracker = YOLOManager(yolo_name='yolov8n_openvino_model/')
        self.detect_isolation = DBScanManager(eps=150)
        self.face_detection = AnalysisFaceManager(
            providers=['OpenVINOExecutionProvider','CPUExecutionProvider'],
            use_gpu=False
        )
        self.emotions = EmotionDetecManager()
        self.persons = {}

        self.face_cache: dict[int, bool | None] = {}

        self.frame_count: int = 0
        self.pass_frame: float = 5

        self.cached_isolations: list = []

        self.face_coldown: float = 5.0

        self.last_face_check: dict[int, float] = {}

    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('nao foi possivel iniciar a leitura')
                break

            frame = cv2.resize(frame, (960, 640))
            self.frame_count += 1

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

                    # print(f'{self.persons}\n')

                if self.persons:
                    if self.frame_count % self.pass_frame == 0:
                        self.cached_isolations = self.detect_isolation.execute(
                            self.persons
                        )

                    lonely_ids = self.cached_isolations

                    if lonely_ids:
                        for person_id in lonely_ids:
                            if person_id not in ids:
                                continue

                            box_ids = self.persons[person_id]['box']

                            if self.cooldown_rechead(person_id):
                                if self.face_cache.get(person_id):
                                    crop = self.cut_frame(frame, box_ids)
                                    cv2.imshow(f'id: {person_id}', crop)
                                    print('verificado')
                                    continue

                            crop = self.cut_frame(frame, box_ids)

                            combined = self.face_detection.get_face_person(
                                'app/assets/arthur.jpeg', crop
                            )

                            if combined is None or combined is False:
                                print('nao e a mesma pessoa')
                                continue

                            self.face_cache[person_id] = bool(combined)

                            self.last_face_check[person_id] = time.time()

                            emocao = self.emotions.capture_emotion(img=crop)
                            print(emocao)

                            cv2.imshow(f'id: {combined}', crop)

                cv2.imshow('EmpatIA', result.plot())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def cooldown_rechead(self, person_id: int):
        now = time.time()
        last_check = self.last_face_check.get(person_id, 0)

        cooldown = (now - last_check) >= self.face_coldown

        return cooldown

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

    def cut_frame(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = [max(0, x1), max(0, y1)]

        crop = frame[y1:y2, x1:x2]
        return crop
