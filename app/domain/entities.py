from dataclasses import dataclass

from app.domain.exceptions import FieldRequiredException


@dataclass
class Person:
    """
    Entidade de Domínio - Person.
    Representa um indivíduo detectado e rastreado no sistema. 
    Esta classe segue os princípios da Clean Architecture, funcionando como uma 
    entidade pura de domínio, isolada de frameworks ou bibliotecas externas.

    - id (int): Identificador único do indivíduo gerado pelo rastreador (YOLO/ByteTrack).
    - boxes (list): Coordenadas espaciais da caixa delimitadora [x1, y1, x2, y2].
    - confidence (float): Grau de certeza (acurácia) da detecção do modelo.

    - change_boxes: Regra de negócio (mutador) que atualiza as coordenadas espaciais 
                    da pessoa após validar defensivamente que os novos dados não estão vazios, 
                    evitando estados inválidos. dispara 'FieldRequiredException' se falhar.
    - to_dict: Mapeador utilitário que converte a entidade em um dicionário nativo, 
               preparando os dados para serialização em JSON antes do envio ao Redis.
    """
    id: int
    boxes: list
    confidence: float

    def change_boxes(self, new_boxes):
        if not new_boxes:
            raise FieldRequiredException('new boxes is required')

        self.boxes = new_boxes

    def to_dict(self):
        return {'id': self.id, 'boxes': self.boxes, 'conf': self.confidence}
