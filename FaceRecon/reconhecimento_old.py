import cv2
import os
import numpy as np
from ultralytics import YOLO  # type: ignore
import threading
import time
import insightface
import pickle
import logging

# ================================
# FUNÇÃO DE CONFIGURAÇÃO DO LOGGER
# ================================
def setup_logger(script_dir):
    """
    Configura dois loggers para salvar em 'historico/escrito'.
    """
    
    # --- CAMINHO PRINCIPAL PARA LOGS ---
    base_log_directory = os.path.join(script_dir, "historico")
    
    # --- CAMINHO PARA OS ARQUIVOS DE TEXTO ---
    text_log_directory = os.path.join(base_log_directory, "escrito")
    
    # --- CAMINHO PARA AS IMAGENS ---
    image_log_directory = os.path.join(base_log_directory, "imagem-nao-aluno")

    # Cria as pastas se não existirem
    os.makedirs(text_log_directory, exist_ok=True)
    os.makedirs(image_log_directory, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- 1. Logger de Alunos (em 'escrito') ---
    logger_alunos = logging.getLogger('AlunosLogger')
    logger_alunos.setLevel(logging.INFO)
    handler_alunos = logging.FileHandler(
        os.path.join(text_log_directory, 'reconhecimento_alunos.log'), 
        mode='a', encoding='utf-8'
    )
    handler_alunos.setFormatter(formatter)
    if not logger_alunos.handlers:
        logger_alunos.addHandler(handler_alunos)
        
    # --- 2. Logger de Alertas (em 'escrito') ---
    logger_alertas = logging.getLogger('AlertasLogger')
    logger_alertas.setLevel(logging.WARNING)
    handler_alertas = logging.FileHandler(
        os.path.join(text_log_directory, 'alertas_nao_alunos.log'), 
        mode='a', encoding='utf-8'
    )
    handler_alertas.setFormatter(formatter)
    if not logger_alertas.handlers:
        logger_alertas.addHandler(handler_alertas)
        
    # Retorna os dois loggers e o caminho para salvar as imagens
    return logger_alunos, logger_alertas, image_log_directory

# ================================
# FUNÇÕES AUXILIARES E CAMINHOS
# ================================
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARQUIVO_BASE_DADOS = os.path.join(SCRIPT_DIR, "base_dados_alunos.pkl")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

MODEL_NAME = "InsightFace" 
DETECTOR_BACKEND = "opencv" 

known_face_embeddings = []
known_face_names = []

try:
    with open(ARQUIVO_BASE_DADOS, 'rb') as file:
        data = pickle.load(file)
        known_face_embeddings = data["embeddings"]
        known_face_names = data["names"]
    print(f"[INFO] Base de dados '{ARQUIVO_BASE_DADOS}' carregada com sucesso com {len(known_face_names)} rostos.")
except FileNotFoundError:
    print(f"[ERRO CRÍTICO] O arquivo '{ARQUIVO_BASE_DADOS}' não foi encontrado. Rode 'cadastro.py' primeiro.")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
print("[INFO] Aplicando configurações de câmera otimizadas...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 115)
cap.set(cv2.CAP_PROP_CONTRAST, 140)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
GAMMA_VALUE = 1.2
model_yolo = YOLO(YOLO_MODEL_PATH)

latest_frame = None
last_known_faces = []
last_known_persons = []
processing_lock = threading.Lock()
is_running = True

# ================================
# LÓGICA DA THREAD DE PROCESSAMENTO
# ================================
def process_frames_insightface(logger_alunos, logger_alertas, image_log_directory):
    global latest_frame, last_known_faces, last_known_persons, is_running

    SIMILARITY_THRESHOLD = 0.52 
    app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    recently_logged = {}
    LOG_COOLDOWN_SECONDS = 1 
    SCALE_FACTOR = 0.5 
    PROCESS_EVERY_N_FRAMES = 5

    frame_count = 0

    while is_running:
        with processing_lock:
            frame_to_process = latest_frame.copy() if latest_frame is not None else None
        if frame_to_process is None: time.sleep(0.01); continue

        frame_count += 1


        if frame_count % PROCESS_EVERY_N_FRAMES == 0:


            small_frame = cv2.resize(frame_to_process, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)


            current_faces = []
            
            try:
                faces = app.get(small_frame)

                for face in faces:
                    live_embedding = face.normed_embedding
                    scores = np.dot(known_face_embeddings, live_embedding)
                    best_match_index = np.argmax(scores)
                    best_score = scores[best_match_index]

                    name = "NAO ALUNO"
                    if best_score > SIMILARITY_THRESHOLD:
                        name = known_face_names[best_match_index]
                    
                    bbox = face.bbox.astype(int)
                    
                    current_time = time.time()
                    if name not in recently_logged or (current_time - recently_logged[name] > LOG_COOLDOWN_SECONDS):
                        recently_logged[name] = current_time
                        
                        if name == "NAO ALUNO":
                            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                            timestamp_ms = f"{timestamp}_{int(current_time * 1000) % 1000}"
                            img_name = f"ALERTA_NAO_ALUNO_{timestamp_ms}.jpg"
                            
                            save_path = os.path.join(image_log_directory, img_name)
                            
                            inverse_scale = 1 / SCALE_FACTOR
                            h_full, w_full = frame_to_process.shape[:2]
                            orig_x1 = max(0, int(bbox[0] * inverse_scale))
                            orig_y1 = max(0, int(bbox[1] * inverse_scale))
                            orig_x2 = min(w_full, int(bbox[2] * inverse_scale))
                            orig_y2 = min(h_full, int(bbox[3] * inverse_scale))
                            
                            cropped_face = frame_to_process[orig_y1:orig_y2, orig_x1:orig_x2].copy() 
                            
                            if cropped_face.size > 0:
                                # --- INÍCIO DAS ALTERAÇÕES PARA TEXTO DINÂMICO E NA PARTE INFERIOR ---
                                h, w, _ = cropped_face.shape

                                # Formata o texto para a imagem
                                time_only = time.strftime("%H:%M:%S")
                                text_on_image = f"ALERTA NAO ALUNO {time_only}" 
                                
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_thickness = 1
                                text_color = (0, 255, 255) # Ciano (BGR)
                                background_color = (0, 0, 0) # Preto

                                # Calcular o font_scale dinamicamente
                                # Começamos com um font_scale base e ajustamos
                                font_scale = 0.6 
                                (text_width, text_height), baseline = cv2.getTextSize(text_on_image, font, font_scale, font_thickness)

                                # Se o texto for muito grande para a largura da imagem, reduzimos o font_scale
                                if text_width > w - 10: # 10 pixels de padding total
                                    font_scale = (w - 10) / text_width * font_scale
                                    (text_width, text_height), baseline = cv2.getTextSize(text_on_image, font, font_scale, font_thickness)
                                    
                                # Posição para o texto (inferior, centralizado horizontalmente)
                                text_x = (w - text_width) // 2 
                                text_y = h - 5 # 5 pixels de padding do fundo

                                # Adiciona um retângulo de fundo (sempre para melhor legibilidade)
                                cv2.rectangle(cropped_face, (text_x - 2, text_y - text_height - 2), 
                                              (text_x + text_width + 2, text_y + baseline + 2), background_color, -1)
                                
                                # Adiciona o texto à imagem
                                cv2.putText(cropped_face, text_on_image, (text_x, text_y), 
                                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                                # --- FIM DAS ALTERAÇÕES ---

                                cv2.imwrite(save_path, cropped_face)
                                logger_alertas.warning(f"ALERTA: Pessoa não cadastrada detectada. Imagem salva em: {save_path}")
                            else:
                                logger_alertas.warning(f"ALERTA: Pessoa não cadastrada detectada. Falha ao salvar imagem (rosto muito pequeno).")
                        else:
                            logger_alunos.info(f"RECONHECIDO: {name}")
                    
                    color = (0, 255, 0) if name != "NAO ALUNO" else (0, 0, 255)
                    current_faces.append((bbox, name, color))



            except Exception as e:
                print(f"[ERRO NA THREAD DE PROCESSAMENTO]: {e}")
            
            results = model_yolo(small_frame, classes=[0], verbose=False)
            current_persons = [box.xyxy[0].numpy().astype(int) for r in results for box in r.boxes]


            with processing_lock:
                last_known_faces = current_faces
                last_known_persons = current_persons

if __name__ == "__main__":
    
    print("[INFO] Configurando sistema de logs...")
    logger_alunos, logger_alertas, image_log_directory = setup_logger(SCRIPT_DIR)
    
    print("🔒 CityLab Security rodando... Pressione 'q' para sair.")
    
    processing_thread = threading.Thread(
        target=process_frames_insightface, 
        args=(logger_alunos, logger_alertas, image_log_directory),
        daemon=True
    )
    processing_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret: is_running = False; break
        frame_final = adjust_gamma(frame, gamma=GAMMA_VALUE)


        with processing_lock: 
            latest_frame = frame_final.copy()

            faces_to_draw = list(last_known_faces)
            persons_to_draw = list(last_known_persons)

        inverse_scale = 1 / 0.5


        for box, name, color in faces_to_draw:

            x1, y1, x2, y2 = [int(coord * inverse_scale) for coord in box]
            cv2.rectangle(frame_final, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_final, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for box in persons_to_draw:
            x1, y1, x2, y2 = [int(coord * inverse_scale) for coord in box]
            is_suspect = False
            if is_suspect:
                cv2.rectangle(frame_final, (x1, y1), (x2, y2), (0, 140, 255), 3)
                cv2.putText(frame_final, "SUSPEITO", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)

        cv2.imshow("CityLab Security", frame_final)
        if cv2.waitKey(1) & 0xFF == ord('q'): is_running = False; break

    print("Encerrando...")
    processing_thread.join()
    cap.release()