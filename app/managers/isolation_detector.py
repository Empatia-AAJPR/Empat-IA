from numpy import array
from sklearn.cluster import DBSCAN


class DBScanManager:
    def __init__(self, eps: float = float(150)) -> None:
        """
        Inicializa o algoritmo DBSCAN configurando a distância
        máxima (eps) e o número mínimo de pessoas para formar um grupo (min_samples=2).
        """
        self.scan = DBSCAN(eps=eps, min_samples=2)
        self.inf_persons: dict
        self.persons_cord: list

    def _to_list(self, inf: dict):
        """
        Extrai as caixas delimitadoras (bounding boxes)
        do dicionário de pessoas para construir uma lista geométrica pura.
        """
        return [p['box'] for p in inf.values()]

    def _to_array(self, model: list):
        """
        Converte a lista de coordenadas em uma matriz NumPy (array),
        que é o formato de dados exigido pelo Scikit-Learn.
        """
        return array(model)

    def find_by_lonely_in_peoples(self, predictions):
        """
        Analisa os rótulos gerados pelo DBSCAN.
        O algoritmo classifica pontos isolados (ruídos de densidade) como -1.
        Este método captura esses índices e retorna os IDs das pessoas isoladas.
        """
        keys = list(self.inf_persons.keys())

        lonelys = []
        for i, value in enumerate(predictions):
            if value == -1:
                lonelys.append(keys[i])

        return lonelys

    def execute(self, persons):
        """
        Recebe o dicionário de pessoas do YOLO,
        transforma os dados em matrizes matemáticas, aplica o algoritmo DBSCAN
        e retorna quem está em situação de isolamento social no ambiente.
        """
        boxes = self._to_list(persons)

        self.persons_cord = boxes
        self.inf_persons = persons

        boxes = self._to_array(boxes)

        clusters = self.scan.fit_predict(boxes)

        return self.find_by_lonely_in_peoples(clusters)
