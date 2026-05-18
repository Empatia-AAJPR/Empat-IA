from fer.fer import FER


class EmotionDetecManager:
    def __init__(self) -> None:
        self.detector = FER(mtcnn=False)

        self.emocoes_ruins = ['sad', 'angry', 'fear', 'disgust']
    def capture_emotion(self, img):
        _img = img

        if _img is None:
            return

        classification = self.detector.detect_emotions(_img)

        return self.classificate_emotions(classification)

    def classificate_emotions(self, emotions: list):
        if not emotions:
            return
        
        _emotions: dict = emotions[0]['emotions']

        conf = max(list(_emotions.values()))

        for key, value in _emotions.items():
            if value == conf:
                return key

    def is_negative_emotion(self, emotion_name: str) -> bool:
        
        if emotion_name in self.emocoes_ruins:
            return True
        return False