import os
import pickle
import cv2
import insightface
import numpy as np

if __name__ == "__main__":
    # --- CONFIGURAÇÕES ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DIR_ALUNOS = os.path.join(PROJECT_ROOT, "alunos")
    ARQUIVO_BASE_DADOS = os.path.join(SCRIPT_DIR, "base_dados_alunos.pkl")

    print("--- INICIANDO PROCESSO DE CADASTRO DE ROSTOS (com InsightFace) ---")

    # Inicializa o modelo InsightFace. Ele fará o download dos modelos na primeira vez.
    # Usamos 'CPUExecutionProvider' para garantir que rode na CPU.
    app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] Modelo InsightFace carregado.")

    known_face_embeddings = []
    known_face_names = []

    if os.path.isdir(DIR_ALUNOS):
        for filename in os.listdir(DIR_ALUNOS):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(DIR_ALUNOS, filename)
                try:
                    # Carrega a imagem com OpenCV
                    img = cv2.imread(path)
                    # 'app.get' faz a detecção e extração do embedding de uma só vez
                    faces = app.get(img)
                    
                    if faces and len(faces) == 1:
                        # Usamos o 'normed_embedding' que é a assinatura facial
                        known_face_embeddings.append(faces[0].normed_embedding)
                        known_face_names.append(os.path.splitext(filename)[0])
                        print(f"[SUCESSO] ✅ - Rosto de '{os.path.splitext(filename)[0]}' cadastrado.")
                    elif not faces:
                        print(f"[FALHA]   ❌ - Nenhum rosto encontrado em '{filename}'.")
                    else:
                        print(f"[FALHA]   ❌ - Múltiplos rostos encontrados em '{filename}'. Apenas um é permitido.")
                except Exception as e:
                    print(f"[ERRO]    ⚠️ - Erro ao processar '{filename}': {e}")
    
    if known_face_embeddings:
        data = {"embeddings": np.array(known_face_embeddings), "names": known_face_names}
        with open(ARQUIVO_BASE_DADOS, 'wb') as file:
            pickle.dump(data, file)
        print(f"\n[SUCESSO] Base de dados salva em '{ARQUIVO_BASE_DADOS}' com {len(known_face_names)} rostos.")
    else:
        print("\n[ERRO] Nenhum rosto pôde ser cadastrado. A base de dados não foi criada.")

    print("--- PROCESSO DE CADASTRO CONCLUÍDO ---")