import cv2
import os
import numpy as np
from ultralytics import YOLO
import threading
import time
from deepface import DeepFace
import multiprocessing # Importamos a nova biblioteca

# ================================
# FUNÇÕES AUXILIARES
# ================================
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# =================================================================
# FUNÇÃO DE GERAÇÃO DE EMBEDDING (PARA SER USADA EM PROCESSOS SEPARADOS)
# =================================================================
def generate_embedding(image_path, model_name, detector_backend):
    """
    Esta função será executada em um processo separado e isolado para cada imagem.
    Isso garante que não haja contaminação de estado entre as chamadas.
    """
    try:
        embedding_obj = DeepFace.represent(img_path=image_path, 
                                           model_name=model_name, 
                                           enforce_detection=True, 
                                           detector_backend=detector_backend)
        # Retorna o embedding se for bem-sucedido
        if embedding_obj and len(embedding_obj) > 0:
            return embedding_obj[0]["embedding"]
    except ValueError:
        # Retorna None se nenhum rosto for encontrado
        return None
    return None

# O restante do código precisa estar dentro deste bloco para o multiprocessing funcionar
if __name__ == "__main__":
    multiprocessing.freeze_support() # Necessário para compatibilidade com Windows

    # ================================
    # INICIALIZAÇÃO DA CÂMERA E CONFIGURAÇÕES FIXAS
    # ================================
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

    # =================================================================
    # CARREGAMENTO DE MODELOS E ROSTOS (VERSÃO MULTIPROCESSING - ROBUSTA)
    # =================================================================
    DIR_ALUNOS = "alunos"
    MODEL_NAME = "Facenet512" 
    DETECTOR_BACKEND = "mtcnn"

    known_face_embeddings = []
    known_face_names = []

    print(f"[INFO] Carregando rostos conhecidos com o modelo '{MODEL_NAME}' e detector '{DETECTOR_BACKEND}'...")
    
    image_paths = []
    names_to_load = []
    if os.path.isdir(DIR_ALUNOS):
        for filename in os.listdir(DIR_ALUNOS):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(DIR_ALUNOS, filename)
                image_paths.append(path)
                names_to_load.append(os.path.splitext(filename)[0])

    if image_paths:
        # Usamos um "Pool" de processos para executar a função 'generate_embedding' para cada imagem
        # O 'starmap' é usado para passar múltiplos argumentos para a nossa função
        args = [(path, MODEL_NAME, DETECTOR_BACKEND) for path in image_paths]
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(generate_embedding, args)
        
        # Filtramos os resultados que falharam (retornaram None)
        for name, embedding in zip(names_to_load, results):
            if embedding is not None:
                known_face_embeddings.append(embedding)
                known_face_names.append(name)
                print(f"[SUCESSO] ✅ - Rosto de '{name}' carregado.")
            else:
                print(f"[FALHA]   ❌ - Não foi possível processar a imagem para '{name}'.")

    print(f"\n[INFO] {len(known_face_embeddings)} rostos carregados com sucesso na base de dados.\n")

    model_yolo = YOLO("yolov8n.pt")

    # ================================
    # LÓGICA DA THREAD DE PROCESSAMENTO
    # ================================
    latest_frame = None
    last_known_faces = []
    last_known_persons = []
    processing_lock = threading.Lock()
    is_running = True

    def find_cosine_distance(source_representation, test_representation):
        a, b = np.asarray(source_representation), np.asarray(test_representation)
        dot, norm = np.dot(a, b), np.linalg.norm(a) * np.linalg.norm(b)
        return 1 - (dot / norm) if norm != 0 else float('inf')

    def process_frames_deepface():
        global latest_frame, last_known_faces, last_known_persons, is_running
        DISTANCE_THRESHOLD = 0.80
        while is_running:
            if not known_face_embeddings:
                time.sleep(2)
                continue
            with processing_lock:
                frame_to_process = latest_frame.copy() if latest_frame is not None else None
            if frame_to_process is None:
                time.sleep(0.01)
                continue
            current_faces = []
            try:
                faces_detected = DeepFace.extract_faces(img_path=frame_to_process, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
                for face_info in faces_detected:
                    if face_info['confidence'] > 0.90:
                        embedding = DeepFace.represent(img_path=face_info['face'], model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
                        min_distance, best_match_name = float('inf'), "NAO ALUNO"
                        for i, known_embedding in enumerate(known_face_embeddings):
                            distance = find_cosine_distance(embedding, known_embedding)
                            if distance < min_distance:
                                min_distance = distance
                                if distance < DISTANCE_THRESHOLD:
                                    best_match_name = known_face_names[i]
                        box_dict = face_info['facial_area']
                        box = (box_dict['x'], box_dict['y'], box_dict['x'] + box_dict['w'], box_dict['y'] + box_dict['h'])
                        color = (0, 255, 0) if best_match_name != "NAO ALUNO" else (0, 0, 255)
                        current_faces.append((box, best_match_name, color))
            except Exception:
                pass
            results = model_yolo(frame_to_process, classes=[0], verbose=False)
            current_persons = [box.xyxy[0].numpy().astype(int) for r in results for box in r.boxes]
            with processing_lock:
                last_known_faces, last_known_persons = current_faces, current_persons

    # ================================
    # INICIALIZAÇÃO DA THREAD E LOOP PRINCIPAL
    # ================================
    print("🔒 CityLab Security rodando... Pressione 'q' para sair.")
    processing_thread = threading.Thread(target=process_frames_deepface, daemon=True)
    processing_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret: is_running = False; break
        frame_final = adjust_gamma(frame, gamma=GAMMA_VALUE)
        with processing_lock: latest_frame = frame_final.copy()
        faces_to_draw, persons_to_draw = list(last_known_faces), list(last_known_persons)
        
        for box, name, color in faces_to_draw:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_final, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_final, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for box in persons_to_draw:
            x1, y1, x2, y2 = box
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