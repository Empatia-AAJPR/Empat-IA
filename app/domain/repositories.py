from abc import ABC, abstractmethod


class IRedisRepository(ABC):
    """
    Interface de Domínio - IRedisRepository.
    Contrato abstrato que dita quais operações de banco de dados Redis devem 
    estar disponíveis para o sistema. Por ser uma interface, não possui código 
    real, servindo para obrigar a classe concreta a implementar todos os métodos listados.
    
    - insert_h: Salva uma informação (campo/valor) dentro de uma tabela Hash do Redis.
    - get_h: Busca uma informação específica de dentro de uma Hash usando o nome do campo.
    - insert_ex: Salva uma informação definindo um tempo de vida em segundos (ex). 
                 Quando esse tempo acaba, o Redis apaga o dado sozinho (usado nos alertas).
    - h_get_all: Puxa tudo o que estiver salvo dentro de uma tabela Hash de uma vez só 
                 (usado para trazer todos os alunos cadastrados).
    """
    @abstractmethod
    def insert_h(self, key: str, field: str, value: str):
        ...

    @abstractmethod
    def get_h(self, key: str, field: str):
        ...
 
    @abstractmethod
    def insert_ex(self, key: str, field: str, value: str, ex: int):
        ...

    @abstractmethod
    def h_get_all(self, key) -> dict:
        ...

