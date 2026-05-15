from numpy import array
from sklearn.cluster import DBSCAN

d = {
    1: {'box': [2.21, 303.94, 212.21, 635.18], 'conf': 0.94},
    2: {'box': [716.72, 508.85, 877.53, 638.26], 'conf': 0.72},
    3: {'box': [702.53, 451.47, 762.8, 555.75], 'conf': 0.42},
    4: {'box': [196.63, 467.38, 356.77, 637.05], 'conf': 0.73},
    5: {'box': [498.08, 459.29, 608.73, 532.28], 'conf': 0.48},
    9: {'box': [486.22, 458.45, 592.54, 532.1], 'conf': 0.45},
    12: {'box': [480.76, 454.85, 592.34, 532.1], 'conf': 0.53},
    13: {'box': [702.4, 450.73, 763.05, 553.96], 'conf': 0.45},
    14: {'box': [693.9, 452.03, 846.03, 547.84], 'conf': 0.46},
}

"""
boxes = array([pessoa['box'] for pessoa in d.values()])

scan = DBSCAN(eps=250, min_samples=2)

predict = scan.fit_predict(boxes)

print(predict)
"""


# ---------------------------------------------


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


if __name__ == '__main__':
    scan = DBScanManager()
    print(scan.execute(persons=d))
