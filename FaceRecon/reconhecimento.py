import cv2
import os
import numpy as np
from ultralytics import YOLO
import threading
import time
import insightface
import pickle

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARQUIVO_BASE_DADOS = os.path.join(SCRIPT_DIR, "base_dados_alunos.pkl")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

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

cap = cv2.VideoCapture(1)
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

def process_frames_insightface():
    global latest_frame, last_known_faces, last_known_persons, is_running
    
    SIMILARITY_THRESHOLD = 0.5 
    app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # --- NOVAS VARIÁVEIS DE OTIMIZAÇÃO ---
    # Fator de escala para a imagem de análise. 0.5 = metade da resolução (4x mais rápido)
    SCALE_FACTOR = 0.5 
    # Processar a análise pesada a cada N frames.
    PROCESS_EVERY_N_FRAMES = 10
    frame_count = 0

    while is_running:
        with processing_lock:
            frame_to_process = latest_frame.copy() if latest_frame is not None else None
        if frame_to_process is None: time.sleep(0.01); continue

        frame_count += 1
        
        # --- BLOCO DE PROCESSAMENTO PESADO (SÓ RODA A CADA N FRAMES) ---
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            
            # 1. Redimensiona a imagem para análise
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

            # 2. Executa a análise pesada na imagem pequena
            current_faces = []
            try:
                faces = app.get(small_frame) # Análise no frame pequeno
                
                for face in faces:
                    live_embedding = face.normed_embedding
                    scores = np.dot(known_face_embeddings, live_embedding)
                    best_match_index = np.argmax(scores)
                    best_score = scores[best_match_index]
                    
                    name = "NAO ALUNO"
                    if best_score > SIMILARITY_THRESHOLD:
                        name = known_face_names[best_match_index]
                        
                    bbox = face.bbox.astype(int)
                    color = (0, 255, 0) if name != "NAO ALUNO" else (0, 0, 255)
                    current_faces.append((bbox, name, color))
                    
            except Exception: pass
            
            results = model_yolo(small_frame, classes=[0], verbose=False) # Análise no frame pequeno
            current_persons = [box.xyxy[0].numpy().astype(int) for r in results for box in r.boxes]

            # 3. Atualiza os resultados na memória de forma segura
            with processing_lock:
                last_known_faces = current_faces
                last_known_persons = current_persons

if __name__ == "__main__":
    print("🔒 CityLab Security rodando... Pressione 'q' para sair.")
    processing_thread = threading.Thread(target=process_frames_insightface, daemon=True)
    processing_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret: is_running = False; break
        frame_final = adjust_gamma(frame, gamma=GAMMA_VALUE)

        # Atualiza o frame para a thread de processamento
        with processing_lock: 
            latest_frame = frame_final.copy()
            # Pega uma cópia dos últimos resultados para desenhar
            faces_to_draw = list(last_known_faces)
            persons_to_draw = list(last_known_persons)

        # --- DESENHA OS RESULTADOS EM TODOS OS FRAMES (PARA FLUIDEZ) ---
        inverse_scale = 1 / 0.5 # Inverso do SCALE_FACTOR usado na thread
        
        for box, name, color in faces_to_draw:
            # Re-escala as coordenadas do retângulo para o tamanho original do frame
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
    cv2.destroyAllWindows()