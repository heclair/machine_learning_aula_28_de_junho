import streamlit as st
import requests

st.set_page_config(page_title="Detector de Fake News", layout="centered")

st.title("🕵️‍♂️ Detector de Fake News com IA")

st.markdown("Digite abaixo o **título** e o **texto** da notícia para verificar se é falsa ou verdadeira.")

# Entrada de dados
title = st.text_input("Título da notícia")
text = st.text_area("Texto completo da notícia")

# Botão para enviar
if st.button("Analisar"):
    if not title or not text:
        st.warning("⚠️ Por favor, preencha tanto o título quanto o texto.")
    else:
        texto_completo = f"{title} {text}"
        with st.spinner("Analisando com IA..."):
            try:
                response = requests.post(
                    "http://backend:8000/api/classificar-noticia",
                    json={"texto": texto_completo}
                )

                if response.status_code == 200:
                    data = response.json()
                    classificacao = data.get("classificacao", "Erro")
                    confianca = data.get("confianca", 0)
                    data_analisada = data.get("data")
                    explicacao = data.get("explicacao", [])

                    if classificacao in ["Real", "True"]:
                        st.markdown(f"""
                            <div style='border: 2px solid green; padding: 15px; border-radius: 10px; background-color: #f0fff0;'>
                                <h3 style='color: green;'>✅ Classificação: Real</h3>
                                <p><strong>Confiança:</strong> {confianca:.2f}%</p>
                                <p><strong>Data da análise:</strong> {data_analisada}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    elif classificacao in ["Fake", "False"]:
                        st.markdown(f"""
                            <div style='border: 2px solid red; padding: 15px; border-radius: 10px; background-color: #fff0f0;'>
                                <h3 style='color: red;'>❌ Classificação: Fake</h3>
                                <p><strong>Confiança:</strong> {confianca:.2f}%</p>
                                <p><strong>Data da análise:</strong> {data_analisada}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Resultado inesperado: {classificacao}")

                    # Mostrar palavras influentes se houver
                    if explicacao:
                        st.markdown("### 🔍 Palavras influentes na decisão do modelo:")
                        for palavra, peso in explicacao:
                            st.markdown(f"- **{palavra}** → peso: `{peso:.3f}`")

                else:
                    st.error(f"Erro {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Erro ao conectar com a API: {e}")
