from fer.fer import FER
import cv2
import numpy as np


class EmotionDetecManager:
    def __init__(self) -> None:
        self.detector = FER(mtcnn=False)

    def capture_emotion(self, img):
        _img = cv2.imread(img)

        if _img is None:
            return
        
        _img = np.array(img)

        return self.detector.detect_emotions(_img)
