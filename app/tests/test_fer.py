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

        classification = self.detector.detect_emotions(_img)

        return self.classificate_emotions(classification)

    def classificate_emotions(self, emotions: list):
        _emotions: dict = emotions[0]['emotions']

        conf = max(list(_emotions.values()))

        for key, value in _emotions.items():
            if value == conf:
                return key


if __name__ == '__main__':
    detector = EmotionDetecManager()
    pessoa = 'app/assets/rian.jpeg'

    print(detector.capture_emotion(pessoa))
