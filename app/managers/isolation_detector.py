from numpy import array
from sklearn.cluster import DBSCAN


class DBScanManager:
    def __init__(self, eps: float = float(250)) -> None:
        self.scan = DBSCAN(eps=eps, min_samples=2)
        self.inf_persons: dict
        self.persons_cord: list

    def _to_list(self, inf: dict):
        return [p['box'] for p in inf.values()]

    def _to_array(self, model: list):
        return array(model)

    def find_by_lonely_in_peoples(self, predictions):

        keys = list(self.inf_persons.keys())

        lonelys = []
        for i, value in enumerate(predictions):
            if value == -1:
                lonelys.append(keys[i])

        return lonelys

    def execute(self, persons):
        boxes = self._to_list(persons)

        self.persons_cord = boxes
        self.inf_persons = persons

        boxes = self._to_array(boxes)

        clusters = self.scan.fit_predict(boxes)

        return self.find_by_lonely_in_peoples(clusters)
