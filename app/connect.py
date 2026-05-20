from redis import Redis
from config.connection_option import connect_option


class RedisConnectionHandler:
    """
    Gerenciador de Conexão - RedisConnectionHandler.
    Classe responsável por centralizar o ciclo de vida da conexão com o banco de 
    dados Redis. Ela mapeia os parâmetros de rede e autenticação para garantir 
    que uma instância válida do cliente Redis esteja disponível para os repositórios.

    - __host: Endereço do servidor Redis.
    - __port: Porta lógica convertida para inteiro.
    - __decode_responses: Booleano que define se os bytes do banco serão convertidos em strings.
    - __username / __password: Credenciais de autenticação seguras.
    
    - connect: Cria e inicializa a instância de conexão com o cliente Redis 
               utilizando as credenciais fornecidas, retornando o objeto de conexão.
    - get_conn: Recupera a conexão ativa que já foi previamente estabelecida.
    """
    def __init__(self) -> None:
        self.__host = connect_option['HOST']
        self.__port = int(connect_option['PORT'])
        self.__decode_responses = bool(connect_option['DECODE'])
        self.__username = connect_option['USERNAME']
        self.__password = connect_option['PASSWORD']

    def connect(self):
        self.__conn = Redis(
            host='scissors-increase-rain-90411.db.redis.io',
            port=11893,
            decode_responses=True,
            username='default',
            password='hjbIbk8Lk652OqoCkHmc3YxPHNGCOD1m',
        )
        return self.__conn

    def get_conn(self):
        return self.__conn
