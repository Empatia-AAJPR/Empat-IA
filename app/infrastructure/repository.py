from redis import Redis
from typing import Any, Dict
from domain.repositories import IRedisRepository


class RedisRepository(IRedisRepository):
    """
    Repositório Concreto - RedisRepository.
    Implementação real do contrato de dados (IRedisRepository) utilizando a 
    biblioteca oficial 'redis'. Esta classe traduz as regras de negócio do 
    sistema em comandos diretos executados no banco de dados em memória.
    Funcionamento dos Métodos:
    - __init__: Recebe e armazena uma conexão ativa do Redis (__conn).
    - insert_h: Executa o comando HSET para salvar dados (campo/valor) em uma Hash.
    - get_h: Executa o comando HGET para buscar um campo específico de uma Hash.
    - insert_ex: Executa um HSET seguido do comando EXPIRE, salvando o dado e definindo 
                 um tempo de vida em segundos para a chave inteira.
    - h_get_all: Executa o comando HGETALL para recuperar todos os registros de uma 
                 tabela Hash de uma só vez.
    """
    def __init__(self, conn: Redis) -> None:
        self.__conn = conn

    def insert_h(self, key: str, field: str, value: str): 
        self.__conn.hset(key, field, value)

    def get_h(
        self,
        key: str,
        field: str,
    ):
        return self.__conn.hget(key, field)

    def insert_ex(self, key: str, field: str, value: str, ex: int):
        self.__conn.hset(key, field, value)
        self.__conn.expire(key, ex)

    def h_get_all(self, key) -> Dict[Any, Any]:
        return self.__conn.hgetall(key)
