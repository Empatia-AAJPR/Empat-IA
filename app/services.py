from ultralytics import YOLO


class YOLOServices:
    def __init__(self, model_name: str) -> None:
        self.model = YOLO(model_name)

    def export_model(self, _format):
        self.model.export(format=_format)


if __name__ == '__main__':
    services = YOLOServices('yolov8n.pt')
    services.export_model('openvino')
