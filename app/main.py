from adapters.appvision import AppVision

from dotenv import load_dotenv

import os


load_dotenv()


if __name__ == '__main__':
    url_capture = os.getenv('url_capture', 1)
    url_capture = url_capture if url_capture != '0' else int(url_capture)

    app = AppVision(url_capture=url_capture)
    app.run()
#'http://192.168.1.244:8080/video'