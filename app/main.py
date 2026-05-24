"""
Ponto de Entrada Principal - Main Script.
Este é o script de inicialização de todo o sistema.
Ele carrega as configurações de ambiente, trata dinamicamente a origem do
feed de vídeo (webcam local ou fluxo de rede/IP) e instancia o orquestrador
AppVision para iniciar o monitoramento em tempo real.

1. Carrega o arquivo oculto '.env' contendo a variável 'url_capture'.
2. Trata a variável 'url_capture': se for um número de índice (0, 1, 2),
   converte para inteiro (ativando a webcam local via OpenCV). Se for uma
   string (como uma URL de câmera IP ou caminho de arquivo de vídeo), mantém como string.
3. Instancia a classe orquestradora 'AppVision' injetando a fonte de captura tratada.
4. Executa o método '.run()' para abrir a janela de visualização e iniciar a IA.
"""

from adapters.appvision import AppVision

from dotenv import load_dotenv

import os

load_dotenv()


if __name__ == '__main__':
    url_capture = os.getenv('url_capture', 0)
    url_capture = (
        url_capture if url_capture not in ['0', '1', '2'] else int(url_capture)
    )
    print(url_capture, type(url_capture))
    app = AppVision(url_capture=url_capture)
    app.run()
