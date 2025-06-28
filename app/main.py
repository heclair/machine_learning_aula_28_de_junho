from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

from app.src.services.classificador import classificar_texto
from app.src.services.historico import Historico
from app.src.services.status import obter_status_modelo


app = FastAPI(title="API de Detecção de Fake News", version="1.0")

# Modelo de entrada
class TextoEntrada(BaseModel):
    texto: str

# Rota principal de classificação
@app.post("/api/classificar-noticia")
def classificar_noticia(entrada: TextoEntrada):
    if not entrada.texto.strip():
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")
    resultado, confianca = classificar_texto(entrada.texto)
    Historico.adicionar(entrada.texto, resultado, confianca)
    return {
        "classificacao": resultado,
        "confianca": confianca,
        "data": datetime.now().isoformat()
    }

# Rota para consultar o histórico
@app.get("/api/historico")
def consultar_historico():
    return Historico.listar()

# Rota para status do modelo
@app.get("/api/status")
def status_modelo():
    return obter_status_modelo()