from fer.fer import FER


class EmotionDetecManager:
    def __init__(self) -> None:
        """
        Método Construtor: Inicializa o detector de expressões faciais da biblioteca FER
        e define quais emoções acionam o protocolo de alerta do sistema.
        """
        self.detector = FER(mtcnn=False)
        self.emocoes_ruins = ['sad', 'angry', 'fear', 'disgust', 'neutral']

    def capture_emotion(self, img):
        """
        Captura de Expressão: Recebe o recorte (crop) da face isolada, valida a imagem
        e envia para a inteligência Artificial fazer a predição das porcentagens emocionais.
        """
        _img = img
        if _img is None:
            return
        classification = self.detector.detect_emotions(_img)
        return self.classificate_emotions(classification)

    def classificate_emotions(self, emotions: list):
        """
        Classificador Estatístico: Analisa o dicionário de probabilidades gerado pelo FER,
        encontra o maior valor de confiança (conf) e retorna o nome da emoção predominante.
        """
        if not emotions:
            return
        _emotions: dict = emotions[0]['emotions']

        conf = max(list(_emotions.values()))

        for key, value in _emotions.items():
            if value == conf:
                return key

    def is_negative_emotion(self, emotion_name: str) -> bool:
        """
        Filtro de Relevância: Verifica se o nome da emoção predominante identificada na câmera
        faz parte da lista de monitoramento crítico ('self.emocoes_ruins').
        """
        if emotion_name in self.emocoes_ruins:
            return True
        return False
