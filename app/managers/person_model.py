from ultralytics import YOLO


class YOLOManager:
    def __init__(
        self, yolo_name: str = 'yolov8n.pt', track_name: str = 'bytetrack.yaml'
    ) -> None:
        """
        Carrega a rede neural YOLOv8 configurada para a tarefa 
        de detecção de objetos e define o arquivo de configuração do algoritmo de rastreamento.
        """
        self.model = YOLO(yolo_name, task='detect')
        self.algorithm_track = track_name

    def detect(self, frame) -> list:
        """
        Executa uma inferência simples no frame atual para encontrar 
        objetos, sem associar IDs de rastreamento entre um quadro e outro.
        """
        return self.model.predict(frame)

    def _track_frame(self, frame):
        """
        Detecta e rastreia os indivíduos no vídeo. 
        Utiliza parâmetros otimizados para filtrar apenas pessoas (classe 0) e atribui 
        um ID fixo a cada uma delas enquanto permanecerem na cena.
        """
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
