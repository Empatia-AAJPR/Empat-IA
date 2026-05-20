"""
Configuração de Infraestrutura - Connection Options.
Este script é responsável por centralizar a extração e o mapeamento das credenciais 
de rede necessárias para estabelecer uma conexão com o banco de dados Redis. Ele 
utiliza a biblioteca 'python-dotenv' para carregar variáveis de ambiente a partir 
de um arquivo oculto '.env', garantindo a segurança dos dados sensíveis.

- HOST: Endereço IP ou domínio do servidor Redis (Retorna vazio '' se não configurado).
- PORT: Porta lógica de comunicação com o banco de dados.
- PASSWORD: Chave secreta de autenticação e controle de acesso.
- USERNAME: Usuário para autenticação ACL (Assume 'default' se não especificado).
- DECODE: Flag que define se o cliente Redis deve decodificar as respostas como strings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

connect_option = {
    'HOST': os.getenv('HOST', ''),
    'PORT': os.getenv('PORT', ''),
    'PASSWORD': os.getenv('PASSWORD', ''),
    'USERNAME': os.getenv('_USERNAME', 'default'),
    'DECODE': os.getenv('DECODE', ''),
}
print(connect_option)
