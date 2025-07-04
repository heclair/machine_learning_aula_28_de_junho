import sys
import os
import numpy as np
from lime.lime_text import LimeTextExplainer

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.src.services.classificador import classificar_texto

print("[INFO] Módulos carregados com sucesso.")
print("[INFO] Inicializando wrapper de predição...")

class FakeNewsWrapper:
    def predict_proba(self, textos):
        resultados = []
        for texto in textos:
            print(f"\n[DEBUG] Classificando texto: {texto[:60]}...")
            label, prob = classificar_texto(texto)
            print(f"[DEBUG] Resultado: {label}, Confiança: {prob}%")

            # Transformar para [prob_true, prob_fake]
            if label == "Fake":
                resultados.append([1 - prob/100, prob/100])
            else:
                resultados.append([prob/100, 1 - prob/100])
        return np.array(resultados)

print("[INFO] Criando explicador LIME...")
explainer = LimeTextExplainer(class_names=["True", "Fake"])

# Frase de exemplo
texto = "A água potável causa problemas à saúde humana"
print(f"\n[INFO] Texto para análise: {texto}")

# Gerar explicação
print("[INFO] Gerando explicação LIME...")
exp = explainer.explain_instance(
    texto,
    classifier_fn=FakeNewsWrapper().predict_proba,
    num_features=10,
    num_samples=100  # <= ajuste aqui (padrão é 500)
)


# Mostrar resultado
print("[INFO] Exibindo explicação no navegador...")
exp.show_in_notebook()
