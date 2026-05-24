import json
from domain.repositories import IRedisRepository
from domain.servicies import ICompaterService
from typing import Any
from datetime import datetime
from uuid import UUID


class GetStudentUseCase:
    def __init__(
        self, redis_repo: IRedisRepository, compacter_service: ICompaterService
    ) -> None:
        """
        Caso de Uso-Recuperação de Alunos: Responsável por buscar todas as fotos e nomes
        cadastrados no banco de dados para que a IA possa carregá-los na memória RAM.
        """
        self.redis_repo = redis_repo
        self.compacter_service = compacter_service

    def execute(self):

        # return [
        #     {key: self.compacter_service.decoded_img(value)}
        #     for key, value in self.redis_repo.h_get_all('back:register').items()
        # ]

        student_data = self.redis_repo.h_get_all('back:resgister')
        if not student_data:
            return []
        result = []
        for key, value in student_data.items():
            payload = json.loads(value)

            if payload['vector']:
                payload['vector'] = self.compacter_service.decoded_img(
                    payload['vector']
                )

            result.append(payload)
        return result


class SendProcessStudentUseCase:
    def __init__(self, redis_repo: IRedisRepository):
        """
        Caso de Uso - Envio de Classificação: Responsável por enviar os alertas em tempo real
        de alunos que foram detectados em situação de vulnerabilidade emocional e isolamento.
        """
        self.redis_repo = redis_repo

    def execute(self, id: UUID, name: str, status: Any):

        payload = {
            'id': id,
            'name': name,
            'emotion': status,
            'timestamp': str(datetime.now()),
        }
        data = json.dumps(payload)

        self.redis_repo.insert_ex('ia:classifieds', name, data, 5)
