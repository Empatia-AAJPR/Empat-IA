import json
from domain.repositories import IRedisRepository
from domain.servicies import ICompaterService
from typing import Any
from datetime import datetime


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
        """
        Executa a leitura em massa (HGETALL) da tabela 'register' no Redis, 
        passa o JSON lido pelo serviço de descompactação e decodificação de imagem, 
        e retorna uma lista estruturada de dicionários com os rostos prontos para o InsightFace.
        """
        return [
            {key: self.compacter_service.decoded_img(json.loads(value))}
            for key, value in self.redis_repo.h_get_all('register').items()
        ]


class SendProcessStudentUseCase:
    def __init__(self, redis_repo: IRedisRepository):
        """
        Caso de Uso - Envio de Classificação: Responsável por enviar os alertas em tempo real 
        de alunos que foram detectados em situação de vulnerabilidade emocional e isolamento.
        """
        self.redis_repo = redis_repo

    def execute(self, name: str, status: Any):
        """
        Monta um payload contendo o estado emocional detectado (status) e o carimbo de data/hora atual (time).
        Converte esses dados para uma string JSON e os insere na tabela 'classifieds' do Redis 
        com um tempo de expiração (TTL) automático de 5 segundos.
        """
        payload = {'status': status, 'time': str(datetime.now())}
        data = json.dumps(payload)

        self.redis_repo.insert_ex('classifieds', name, data, 5)
