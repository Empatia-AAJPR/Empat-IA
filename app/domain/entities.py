from dataclasses import dataclass

from app.domain.exceptions import FieldRequiredException


@dataclass
class Person:
    id: int
    boxes: list
    confidence: float

    def change_boxes(self, new_boxes):
        if not new_boxes:
            raise FieldRequiredException('new boxes is required')

        self.boxes = new_boxes

    def to_dict(self):
        return {'id': self.id, 'boxes': self.boxes, 'conf': self.confidence}
