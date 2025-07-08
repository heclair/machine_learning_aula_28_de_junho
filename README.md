# 📰 API de Detecção de Fake News (PT-BR)

Este projeto utiliza **BERTimbau** e **Logistic Regression** para classificar textos curtos em português do Brasil como **notícias verdadeiras ou falsas**, com foco especial em frases sensíveis que envolvem ética, saúde e diversidade.

---

## 🚀 Funcionalidades

- 🔎 Classificação de frases como **Fake** ou **True**
- 📈 Retorno da **confiança (%)**
- 💬 Explicação com as palavras mais influentes (**LIME**)
- 🧠 Suporte a frases curtas e sensíveis (ex: "água faz mal à saúde")
- 🕓 Histórico de classificações realizadas
- ⚙️ Backend com **FastAPI**
- 🌐 Interface com **Streamlit**
- 📦 Conteinerização com **Docker**
- ☸️ Deploy com **Kubernetes (Minikube)**

---

## 📁 Estrutura de Pastas

```
machine_learning_aula_28_de_junho/
├── app/
│   ├── main.py                  # API principal (FastAPI)
│   ├── src/                     # Lógica da API: classificação, status, histórico
│   ├── data/                    # Vetores TF-IDF (ignorado no Git)
│   ├── data_bert/               # Embeddings BERT e modelos treinados
│   ├── Fake.br-Corpus/          # Dataset original e pré-processado
│   └── treino/                  # Scripts de pré-processamento, treinamento e LIME
├── frontend/
│   ├── app.py                   # Interface Streamlit
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile                   # Dockerfile do backend
├── docker-compose.yml           # Compose para backend + frontend
└── README.md                    # Este arquivo
```

---

## ⚙️ Como rodar o projeto localmente com Docker Compose

### 1. Pré-requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Rodar a aplicação

```bash
docker-compose up --build
```

### 3. Acessar os serviços

- 🧠 **Backend (FastAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)  
- 🌐 **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501)

---

## 📡 Endpoints

### `POST /api/classificar-noticia`

Classifica um texto como Fake ou True.

**Exemplo de entrada:**
```json
{
  "texto": "água faz mal à saúde"
}
```

**Resposta esperada:**
```json
{
  "classificacao": "Fake",
  "confianca": 97.32,
  "data": "2025-06-28T12:00:00",
  "explicacao": [["água", 0.42], ["saúde", -0.21]]
}
```

---

### `GET /api/historico`

Retorna o histórico de classificações realizadas.

---

### `GET /api/status`

Retorna o status do modelo carregado (tipo, embeddings, versão).

---

## 🌐 Interface Web (Streamlit)

Permite ao usuário:

- ✅ Digitar o título/texto da notícia
- 🔍 Ver se é Fake ou True
- 📊 Visualizar confiança da classificação
- 💬 Ver explicações LIME (palavras influentes)
- 🕓 Ver a data da análise

---

## ☸️ Como fazer o deploy com Kubernetes

### 1. Build e push das imagens

```bash
# Backend
docker build -t seuusuario/fake-news-backend:latest .
docker push seuusuario/fake-news-backend:latest

# Frontend
docker build -t seuusuario/fake-news-frontend:latest ./frontend
docker push seuusuario/fake-news-frontend:latest
```

### 2. Instalar o Minikube (Windows via PowerShell)

```bash
choco install minikube -y
```

### 3. Iniciar o cluster

```bash
minikube start
```

### 4. Criar recursos no Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 5. Acessar via Minikube

```bash
minikube service fake-news-service
```

---

## 🧠 Modelo

- **Embeddings**: BERTimbau (`neuralmind/bert-base-portuguese-cased`)
- **Classificador**: Regressão Logística
- **Explicabilidade**: LIME (Local Interpretable Model-Agnostic Explanations)
- **Treinamento**: `bert_train.py`
- **Dados**: Fake.Br Corpus + frases sociais/éticas criadas para melhorar a generalização

---

## 📄 Licença

Este projeto é de **uso educacional e acadêmico**. Para uso em produção, é necessário aplicar filtros adicionais, controle de viés e validação contínua.

---

## 🙋‍♂️ Contato

Dúvidas ou sugestões? Contribuições são bem-vindas!
