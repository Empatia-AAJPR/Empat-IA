from abc import ABC, abstractmethod


class IAnalysisFaceManager(ABC):
    @abstractmethod
    def parser_images(self, images: list) -> None:
        ...


class IYOLOManager(ABC):
    @abstractmethod
    def detect(self, frame) -> list:
        ...

    @abstractmethod
    def _track_frame(self, frame):
        ...


class IAppVision(ABC):
    @abstractmethod
    def run(self) -> None:
        ...
