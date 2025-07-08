# 📰 API de Detecção de Fake News (PT-BR)

Este projeto utiliza **BERTimbau** e **Logistic Regression** para classificar textos curtos em português do Brasil como **notícias verdadeiras ou falsas**, com foco especial em frases sensíveis e curtas que envolvem ética, saúde e diversidade.

---

## 🚀 Funcionalidades

- 🔎 Classificação de frases como **Fake** ou **True**
- 📈 Retorno da **confiança (%)**
- 💬 Explicação das palavras mais influentes (**LIME**)
- 💬 Suporte a frases curtas e sensíveis (ex: "água faz mal à saúde")
- 📚 Histórico de classificações realizadas
- ⚙️ Backend com **FastAPI**
- 🌐 Interface com **Streamlit**
- 📦 Conteinerização com **Docker**
- ☸️ Deploy com **Kubernetes (Minikube)**

---

## 📁 Estrutura de Pastas

machine_learning_aula_28_de_junho/
├── app/
│ ├── main.py # API principal (FastAPI)
│ ├── src/ # Lógica da API: classificação, status, histórico
│ ├── data/ # Vetores TF-IDF (ignorado no Git)
│ ├── data_bert/ # Embeddings BERT e modelos treinados
│ ├── Fake.br-Corpus/ # Dataset original e pré-processado
│ └── treino/ # Scripts de pré-processamento, treinamento e LIME
├── frontend/ # Interface em Streamlit
│ ├── app.py
│ └── requirements.txt
├── k8s/ # Manifests do Kubernetes
│ ├── deployment.yaml
│ └── service.yaml
├── Dockerfile # Backend Dockerfile
├── docker-compose.yml # Compose para backend + frontend
└── README.md # Este arquivo


---

## ⚙️ Como rodar o projeto localmente com Docker Compose

### 1. Pré-requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Rodar aplicação

```bash
docker-compose up --build

3. Acessar os serviços
🧠 Backend: http://localhost:8000/docs

🌐 Frontend: http://localhost:8501


📡 Endpoints
POST /api/classificar-noticia
Classifica um texto como Fake ou True.

{
  "texto": "água faz mal à saúde"
}


{
  "classificacao": "Fake",
  "confianca": 97.32,
  "data": "2025-06-28T12:00:00",
  "explicacao": [["água", 0.42], ["saúde", -0.21]]
}

GET /api/historico
Retorna o histórico de classificações realizadas.

GET /api/status
Retorna status atual do modelo carregado (tipo, embeddings, versão).

🌐 Interface Web (Streamlit)
Permite ao usuário digitar título e texto da notícia para:

✅ Exibir se é Fake ou True

📊 Mostrar confiança (%)

💬 Exibir explicações LIME (palavras influentes)

🕓 Mostrar data da análise

📦 Como fazer o deploy com Kubernetes

docker build -t seuusuario/fake-news-backend:latest .
docker push seuusuario/fake-news-backend:latest

docker build -t seuusuario/fake-news-frontend:latest ./frontend
docker push seuusuario/fake-news-frontend:latest

2. Instalar o Minikube (Windows via PowerShell)

choco install minikube -y

3. Iniciar o cluster

minikube start

4. Criar recursos no Kubernetes

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

5. Acessar via Minikube

minikube service fake-news-service


🧠 Modelo
Embeddings: BERTimbau (neuralmind/bert-base-portuguese-cased)

Classificador: LogisticRegression

Explicabilidade: LIME (Local Interpretable Model-agnostic Explanations)

Treinamento: bert_train.py

Dados: Corpus Fake.Br + frases sociais e éticas criadas para melhorar a generalização

📄 Licença
Este projeto é de uso educacional e acadêmico. Para uso em produção, é importante aplicar filtros adicionais, controle de viés e validação contínua.

🙋‍♂️ Contato
Dúvidas ou sugestões? Contribuições são bem-vindas!