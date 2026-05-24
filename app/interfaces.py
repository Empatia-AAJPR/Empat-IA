from abc import ABC, abstractmethod


class IAnalysisFaceManager(ABC):
    """
    Interface de Domínio - IAnalysisFaceManager.
    Contrato para o gerenciador de análise e reconhecimento facial. Define a
    operação necessária para processar e extrair características ou embeddings
    de um conjunto de imagens de rostos.
    """

    @abstractmethod
    def parser_images(self, images: list) -> None:
        ...


class IYOLOManager(ABC):
    """
    Interface de Domínio - IYOLOManager.
    Contrato para o modelo de visão computacional (YOLO). Define os métodos
    essenciais para a localização espacial pura de objetos e para o
    rastreamento contínuo e persistente de indivíduos ao longo dos frames.
    """

    @abstractmethod
    def detect(self, frame) -> list:
        ...

    @abstractmethod
    def _track_frame(self, frame):
        ...


class IAppVision(ABC):
    """
    Interface de Domínio - IAppVision.
    Contrato para a aplicação orquestradora principal (AppVision). Define o
    ponto de entrada obrigatório para iniciar a captura do feed de vídeo e
    executar o loop principal do sistema.
    """

    @abstractmethod
    def run(self) -> None:
        ...
