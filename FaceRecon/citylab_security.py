import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO

# ================================
# Diretórios e Modelos (sem alterações)
# ================================
DIR_ALUNOS = "alunos"

known_face_encodings = []
known_face_names = []

print("[INFO] Carregando rostos conhecidos...")
for filename in os.listdir(DIR_ALUNOS):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DIR_ALUNOS, filename)
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

print(f"[INFO] {len(known_face_encodings)} rostos carregados.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = YOLO("yolov8n.pt")

# ================================
# Inicializar câmera (sem alterações)
# ================================
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

print("🔒 CityLab Security rodando... Pressione 'q' para sair.")

# ===============================================
# NOVAS VARIÁVEIS PARA OTIMIZAÇÃO (TÉCNICAS 1 E 3)
# ===============================================
# Processar a cada N frames. Aumente este valor se o vídeo ainda travar.
# Um bom valor para começar sem redimensionar é entre 10 e 15.
PROCESS_EVERY_N_FRAMES = 10
frame_count = 0

# Listas para armazenar os resultados do último frame processado
# Isso é crucial para a Técnica 3: manter os resultados
last_known_faces = []
last_known_persons = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # =========================================================================
    # BLOCO DE PROCESSAMENTO PESADO - SÓ EXECUTA A CADA N FRAMES (TÉCNICA 1)
    # =========================================================================
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Limpa os resultados antigos antes de processar novamente
        last_known_faces = []
        last_known_persons = []

        # Converte o frame para os formatos necessários (cinza e RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- RECONHECIMENTO FACIAL ---
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            name = "NAO ALUNO"
            color = (0, 0, 255)

            face_locations = [(y, x + w, y + h, x)]
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if encodings:
                matches = face_recognition.compare_faces(known_face_encodings, encodings[0], tolerance=0.6)
                if True in matches:
                    idx = matches.index(True)
                    name = known_face_names[idx]
                    color = (0, 255, 0)
            
            # Salva o resultado (bounding box, nome, cor) na lista
            box = (x, y, x+w, y+h)
            last_known_faces.append((box, name, color))

        # --- DETECÇÃO DE PESSOAS COM YOLO ---
        results = model(frame, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                # Salva o bounding box do YOLO na lista
                last_known_persons.append(box.xyxy[0].numpy().astype(int))

    # ===============================================================
    # BLOCO DE DESENHO - EXECUTA EM TODOS OS FRAMES (TÉCNICA 3)
    # Desenha os resultados da última análise bem-sucedida.
    # ===============================================================
    
    # Desenha os retângulos de faces
    for box, name, color in last_known_faces:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Desenha os retângulos de pessoas (com lógica de suspeito)
    for box in last_known_persons:
        x1, y1, x2, y2 = box
        is_suspect = False  # Placeholder
        if is_suspect:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
            cv2.putText(frame, "SUSPEITO", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)


    cv2.imshow("CityLab Security", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 