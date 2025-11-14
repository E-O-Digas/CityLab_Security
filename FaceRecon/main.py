import cv2
import numpy as np
import base64
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from reconhecimento import ProcessadorCV

# ================================
# INICIALIZAÇÃO DO SERVIDOR E MODELOS
# ================================
app = FastAPI(title="Servidor de Reconhecimento CityLab")

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Carregando modelos de CV... Isso pode levar um momento.")
processador = ProcessadorCV()
print("[INFO] Modelos carregados. Servidor pronto para conexões.")

@app.get("/")
async def root():
    return {"message": "Servidor de Reconhecimento CityLab está online."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Novo cliente conectado via WebSocket.")
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                header, encoded = data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
            except ValueError:
                print("[ERRO] Formato de Base64 inválido recebido.")
                continue
                
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("[ERRO] Falha ao decodificar a imagem recebida.")
                continue
                
            results = await asyncio.to_thread(processador.processar_frame, frame)

            await websocket.send_json(results)

    except WebSocketDisconnect:
        print("[INFO] Cliente desconectado.")
    except Exception as e:
        print(f"[ERRO FATAL NO WEBSOCKET]: {e}")
        
# ================================
# PONTO DE ENTRADA PRINCIPAL
# ================================
if __name__ == "__main__":
    print("--- Iniciando Servidor FastAPI (main.py) ---")
    print("Acesse em: http://127.0.0.1:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)