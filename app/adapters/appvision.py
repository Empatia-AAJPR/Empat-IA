import cv2

from managers.isolation_detector import DBScanManager
from managers.person_model import YOLOManager


class AppVision:
    def __init__(self, url_capture: str | int) -> None:
        self.cap = cv2.VideoCapture(url_capture)
        self.tracker = YOLOManager(yolo_name='yolov8n_openvino_model/')
        self.detect_isolation = DBScanManager(eps=150)
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

                    # print(f'{self.persons}\n')

                if self.persons:
                    lonely_ids = self.detect_isolation.execute(self.persons)

                    if lonely_ids:
                        for person in lonely_ids:
                            box_ids = self.persons[person]['box']

                            crop = self.cut_frame(frame, box_ids)

                            if person not in ids:
                                continue

                            print(person)
                            cv2.imshow(f"id: {person}", crop)

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

    def cut_frame(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = [max(0, x1), max(0, y1)]

        crop = frame[y1:y2, x1:x2]
        return crop
