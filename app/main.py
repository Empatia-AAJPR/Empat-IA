from adapters.appvision import AppVision

from dotenv import load_dotenv

import os


load_dotenv()


if __name__ == '__main__':
    app = AppVision(url_capture=os.getenv('url_capture', ''))

    app.run()
