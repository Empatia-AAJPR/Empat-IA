import cv2

from ultralytics import YOLO

from insightface.app import FaceAnalysis

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