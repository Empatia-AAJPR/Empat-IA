import os
# Desativa a atualização automática do YOLO para evitar travamentos/mensagens no terminal
os.environ['YOLO_AUTOUPDATE'] = 'False'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import cv2
from application.use_cases import GetStudentUseCase, SendProcessStudentUseCase
from connect import RedisConnectionHandler
from infrastructure.repository import RedisRepository
from infrastructure.services import CompacterService

from managers.emotions import EmotionDetecManager
from managers.face import AnalysisFaceManager
from managers.isolation_detector import DBScanManager
from managers.person_model import YOLOManager


class AppVision:
    def __init__(self, url_capture: str | int) -> None:
        """
        Método Construtor: Inicializa a captura de vídeo e carrega todos os
        modelos de Inteligência Artificial na memória do computador.
        """
        self.cap = cv2.VideoCapture(url_capture)
        self.tracker = YOLOManager(yolo_name='yolov8n.pt')
        self.detect_isolation = DBScanManager(eps=240)
        self.face_detection = AnalysisFaceManager(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            use_gpu=True,
        )
        self.emotions = EmotionDetecManager()
        self.persons = {}
        self.face_cache: dict[int, bool | None] = {}
        self.frame_count: int = 0
        self.pass_frame: float = 5
        self.cached_isolations: list = []
        self.face_coldown: float = 5.0
        self.last_face_check: dict[int, float] = {}
        self.__conn = RedisConnectionHandler().connect()
        self.redis_repo = RedisRepository(self.__conn)
        self.load_registers()

    def load_registers(self):
        """
        Carga Inicial de Cadastros: Conecta ao Redis, recupera os dados dos alunos,
        descompacta as informações e injeta os rostos no gerenciador de faces de uma só vez.
        """
         
        get_student_use_case = GetStudentUseCase(
            self.redis_repo, CompacterService()
        )
        student = get_student_use_case.execute()

        if student:
            self.face_detection.register_faces(student)

    def run(self) -> None:
        """
        Loop Principal: Captura o feed de vídeo quadro a quadro, executa o rastreamento,
        avalia proximidade física, identifica rostos e analisa emoções em tempo real.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                continue

            self.persons = {}

            if self.frame_count % 90 == 0:
                self.load_registers()

            frame = cv2.resize(frame, (960, 640))
            self.frame_count += 1
            results = self.tracker._track_frame(frame)

            for result in results:
                attr = self._get_atributes(result)
                boxes = attr['boxes']
                confs = attr['confidence']
                ids = self._get_ids(result, boxes)

                for box, conf, id in zip(boxes, confs, ids):
                    if id is not None:

                        id = int(id)

                        self.persons[id] = {
                            'box': list(
                                map(lambda x: round(float(x), 2), box)
                            ),
                            'conf': round(float(conf), 2),
                        }

                if self.persons:
                    lonely_ids = self.cached_isolations
                    if self.frame_count % self.pass_frame == 0:
                        self.cached_isolations = self.detect_isolation.execute(
                            self.persons
                        )
                        lonely_ids = self.cached_isolations
                    if lonely_ids:
                        for person_id in lonely_ids:
                            if person_id not in ids:
                                continue

                            box_ids = self.persons[person_id]['box']
                           
                            crop = self.cut_frame(frame, box_ids)

                            if crop.size <= 0:
                                continue
                            combined = self.face_detection.get_face_person(
                                crop
                            )
                            if combined is None or combined is False:
                                continue
                            self.face_cache[person_id] = bool(combined)
                            self.last_face_check[person_id] = time.time()
                            emocao = self.emotions.capture_emotion(img=crop)

                            if not self.emotions.is_negative_emotion(emocao):
                                continue

                            send_process_use_case = SendProcessStudentUseCase(
                                self.redis_repo
                            )
                            print(f'VERIFICACAO: {combined} e {emocao}')

                            if emocao:
                                send_process_use_case.execute(combined, emocao)

                            cv2.imshow('Imagem Verificada', crop)

                        emocao = ''
                cv2.imshow('EmpatIA', result.plot())
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def cooldown_rechead(self, person_id: int):
        """
        Controlador de Tempo: Verifica se o intervalo de tempo (cooldown)
        estipulado já passou para um determinado ID rastreado.
        """
        now = time.time()
        last_check = self.last_face_check.get(person_id, 0)
        cooldown = (now - last_check) >= self.face_coldown
        return cooldown

    def _get_atributes(self, result):
        """
        Função Utilitária: Extrai e converte as caixas geométricas (bounding boxes)
        e as confianças da predição do YOLO para matrizes NumPy legíveis.
        """
        return {
            'boxes': result.boxes.xyxy.cpu().numpy(),
            'confidence': result.boxes.conf.cpu().numpy(),
        }

    def _get_ids(self, result, boxes):
        """
        Função Utilitária: Recupera os IDs únicos gerados pelo rastreador (ByteTrack/BoT-SORT)
        do YOLO para cada indivíduo na cena. Se não houver ID, preenche com None.
        """
        if result.boxes.id is not None:
            return result.boxes.id.cpu().numpy().astype(int)
        else:
            return [None] * len(boxes)

    def cut_frame(self, frame, box):
        """
        Recorte de Imagem (Crop): Utiliza fatiamento de matrizes (Slicing) do NumPy/OpenCV
        para extrair com precisão a sub-imagem correspondente à caixa delimitadora recebida.
        """
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = [max(0, x1), max(0, y1)]
        crop = frame[y1:y2, x1:x2]
        return crop
