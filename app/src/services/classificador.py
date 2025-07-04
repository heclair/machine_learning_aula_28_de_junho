import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from lime.lime_text import LimeTextExplainer

# Caminhos dos modelos
MODEL_PATH = "app/data_bert/bert_logistic_model.joblib"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Carregamento único
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)
bert.eval()
model = joblib.load(MODEL_PATH)

def gerar_embedding(texto: str):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

def classificar_texto(texto: str):
    embedding = gerar_embedding(texto)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][pred]
    resultado = "Fake" if pred == 1 else "True"
    return resultado, round(prob * 100, 2)

import time
print("[INFO] Iniciando explicação do texto...")
inicio = time.time()


def explicar_texto(texto: str, num_palavras: int = 5):
    import time
    print("[INFO] Iniciando explicação do texto...")
    inicio = time.time()

    def predict_fn(textos):
        probs = []
        for t in textos:
            label, prob = classificar_texto(t)
            if label == "Fake":
                probs.append([1 - prob / 100, prob / 100])
            else:
                probs.append([prob / 100, 1 - prob / 100])
        return np.array(probs)

    explainer = LimeTextExplainer(class_names=["True", "Fake"])
    explicacao = explainer.explain_instance(
        texto,
        classifier_fn=predict_fn,
        num_features=num_palavras,
        num_samples=100  # otimizando desempenho
    )

    print(f"[INFO] Explicação concluída em {time.time() - inicio:.2f} segundos.")
    return explicacao.as_list()
