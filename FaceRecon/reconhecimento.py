import cv2
import os
import numpy as np
from ultralytics import YOLO # type: ignore
import time
import insightface
import pickle
import logging

def setup_logger(script_dir):
    """
    Configura dois loggers para salvar em 'historico/escrito'.
    """
    base_log_directory = os.path.join(script_dir, "historico")
    text_log_directory = os.path.join(base_log_directory, "escrito")
    image_log_directory = os.path.join(base_log_directory, "imagem-nao-aluno")

    os.makedirs(text_log_directory, exist_ok=True)
    os.makedirs(image_log_directory, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger_alunos = logging.getLogger('AlunosLogger')
    logger_alunos.setLevel(logging.INFO)
    if not logger_alunos.handlers:
        handler_alunos = logging.FileHandler(
            os.path.join(text_log_directory, 'reconhecimento_alunos.log'), 
            mode='a', encoding='utf-8'
        )
        handler_alunos.setFormatter(formatter)
        logger_alunos.addHandler(handler_alunos)
        
    logger_alertas = logging.getLogger('AlertasLogger')
    logger_alertas.setLevel(logging.WARNING)
    if not logger_alertas.handlers:
        handler_alertas = logging.FileHandler(
            os.path.join(text_log_directory, 'alertas_nao_alunos.log'), 
            mode='a', encoding='utf-8'
        )
        handler_alertas.setFormatter(formatter)
        logger_alertas.addHandler(handler_alertas)
        
    return logger_alunos, logger_alertas, image_log_directory

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

class ProcessadorCV:
    def __init__(self):
        print("[INFO] Inicializando o ProcessadorCV...")
        
        # ... (Configuração de caminhos permanece igual) ...
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(self.SCRIPT_DIR)
        ARQUIVO_BASE_DADOS = os.path.join(self.SCRIPT_DIR, "base_dados_alunos.pkl")
        
        YOLO_MODEL_PATH = os.path.join(self.SCRIPT_DIR, "yolov8n.pt") 
        if not os.path.exists(YOLO_MODEL_PATH):
            YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt") 
            if not os.path.exists(YOLO_MODEL_PATH):
                print("="*50)
                print(f"[ERRO CRÍTICO] O arquivo 'yolov8n.pt' não foi encontrado.")
                raise FileNotFoundError("Modelo YOLO não encontrado em locais esperados.")

        print(f"[INFO] Procurando Base de Dados em: {ARQUIVO_BASE_DADOS}")
        print(f"[INFO] Modelo YOLO encontrado em: {YOLO_MODEL_PATH}")

        # --- Configuração de Constantes de Processamento ---
        self.SIMILARITY_THRESHOLD = 0.52 
        self.SCALE_FACTOR = 0.5
        self.GAMMA_VALUE = 1.2 
        self.LOG_COOLDOWN_SECONDS = 1 
        
        # ... (Loggers, Base de Dados, Modelos permanecem iguais) ...
        print("[INFO] Configurando sistema de logs...")
        self.logger_alunos, self.logger_alertas, self.image_log_directory = setup_logger(self.SCRIPT_DIR)
        
        print("[INFO] Carregando base de dados .pkl...")
        self.known_face_embeddings = []
        self.known_face_names = []
        try:
            with open(ARQUIVO_BASE_DADOS, 'rb') as file:
                data = pickle.load(file)
                self.known_face_embeddings = data["embeddings"]
                self.known_face_names = data["names"]
            print(f"[INFO] Base de dados carregada com {len(self.known_face_names)} rostos.")
        except FileNotFoundError:
            print(f"[AVISO] O arquivo '{ARQUIVO_BASE_DADOS}' não foi encontrado.")
            self.logger_alertas.warning(f"Arquivo da base de dados não encontrado: {ARQUIVO_BASE_DADOS}")
        
        print("[INFO] Carregando modelo YOLO...")
        self.model_yolo = YOLO(YOLO_MODEL_PATH)
        
        print("[INFO] Carregando modelo InsightFace...")
        self.app_insight = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app_insight.prepare(ctx_id=0, det_size=(640, 640))
        
        self.recently_logged = {}
        print("[INFO] ProcessadorCV inicializado e pronto.")

    def processar_frame(self, frame_to_process):
        current_faces_results = []
        current_persons_results = []

        try:            
            frame_ajustado = adjust_gamma(frame_to_process, gamma=self.GAMMA_VALUE)          
            small_frame = cv2.resize(frame_ajustado, (0, 0), fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR)
            faces = self.app_insight.get(small_frame) 

            if not faces:
                print("[DEBUG] Nenhum rosto detectado neste frame.")

            for face in faces:
                live_embedding = face.normed_embedding
                
                if len(self.known_face_embeddings) == 0:
                    name = "NAO ALUNO"
                    best_score = 0.0
                else:
                    scores = np.dot(self.known_face_embeddings, live_embedding)
                    best_match_index = np.argmax(scores)
                    best_score = scores[best_match_index]
                    
                    print(f"[DEBUG] Rosto detectado. Melhor score: {best_score:.2f} (Threshold: {self.SIMILARITY_THRESHOLD})")

                    name = "NAO ALUNO"
                    if best_score > self.SIMILARITY_THRESHOLD:
                        name = self.known_face_names[best_match_index]
                
                bbox = face.bbox.astype(int)
                
                current_faces_results.append({
                    "name": name,
                    "bbox": [int(coord / self.SCALE_FACTOR) for coord in bbox],
                    "confidence": float(best_score)
                })
                
                # --- Lógica de Log ---
                current_time = time.time()
                if name not in self.recently_logged or (current_time - self.recently_logged[name] > self.LOG_COOLDOWN_SECONDS):
                    self.recently_logged[name] = current_time
                    
                    if name == "NAO ALUNO":
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        timestamp_ms = f"{timestamp}_{int(current_time * 1000) % 1000}"
                        img_name = f"ALERTA_NAO_ALUNO_{timestamp_ms}.jpg"
                        save_path = os.path.join(self.image_log_directory, img_name)
                        
                        inverse_scale = 1 / self.SCALE_FACTOR
                        h_full, w_full = frame_to_process.shape[:2]
                        orig_x1 = max(0, int(bbox[0] * inverse_scale))
                        orig_y1 = max(0, int(bbox[1] * inverse_scale))
                        orig_x2 = min(w_full, int(bbox[2] * inverse_scale))
                        orig_y2 = min(h_full, int(bbox[3] * inverse_scale))
                        
                        cropped_face = frame_to_process[orig_y1:orig_y2, orig_x1:orig_x2].copy() 
                        
                        if cropped_face.size > 0:
                            cv2.imwrite(save_path, cropped_face)
                            self.logger_alertas.warning(f"ALERTA: Pessoa não cadastrada. Imagem salva: {save_path}")
                        else:
                            self.logger_alertas.warning(f"ALERTA: Pessoa não cadastrada. Falha ao salvar (rosto pequeno).")
                    else:
                        self.logger_alunos.info(f"RECONHECIDO: {name}")

            results_yolo = self.model_yolo(small_frame, classes=[0], verbose=False) 
            for r in results_yolo:
                for box in r.boxes:
                    bbox_person = box.xyxy[0].numpy().astype(int)
                    current_persons_results.append({
                        "bbox": [int(coord / self.SCALE_FACTOR) for coord in bbox_person],
                        "confidence": float(box.conf[0])
                    })
                    
        except Exception as e:
            print(f"[ERRO NO PROCESSAMENTO]: {e}")
            self.logger_alertas.error(f"Erro ao processar frame: {e}")

        return {"faces": current_faces_results, "persons": current_persons_results}