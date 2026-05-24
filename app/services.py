from ultralytics import YOLO


class YOLOServices:
    """
    Serviço de Infraestrutura - YOLOServices.
    Classe responsável por gerenciar tarefas utilitárias e operacionais do modelo
    YOLOv8 que ficam fora do pipeline principal de inferência de vídeo, como a
    conversão e exportação de formatos para otimização em diferentes Hardwares.
    - __init__: Inicializa e carrega os pesos do modelo YOLOv8 especificado.
    - export_model: Exporta a rede neural para formatos otimizados de terceiros
                    (como 'openvino' para CPUs Intel, 'onnx', 'engine' para TensorRT, etc.),
                    permitindo ganho de performance e redução de latência em produção.
    """

    def __init__(self, model_name: str) -> None:
        self.model = YOLO(model_name)

    def export_model(self, _format):
        self.model.export(format=_format)


if __name__ == '__main__':
    services = YOLOServices('yolov8n.pt')
    services.export_model('openvino')
