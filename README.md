# Chatbot de Documentação Interna com FastAPI

Este projeto é uma aplicação web completa para gerenciamento e consulta de documentos internos. O backend é construído com FastAPI e o frontend é uma página HTML estática que consome as APIs do backend.

## Funcionalidades

- **API RESTful**: Backend robusto com FastAPI para todas as operações.
- **Chatbot de Perguntas e Respostas**: Responde a perguntas com base em uma base de conhecimento de documentos, acessível via API.
- **Criação e Edição de Documentos**: APIs para criar, padronizar e aprimorar documentos `.docx`.
- **Busca na Web (Simulada)**: Gera conteúdo para novos documentos a partir de uma pergunta, simulando uma busca na internet.
- **Frontend Simples**: Uma interface de usuário em `index.html` para interagir com as funcionalidades do backend.

## Pré-requisitos

- Python 3.9 ou superior
- Git

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO_AQUI>
   cd chatbot_hackathon
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # No Windows
   # source venv/bin/activate  # No macOS/Linux
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**
   - Renomeie o arquivo `.env.example` para `.env` (se ele existir) ou crie um novo.
   - Adicione sua chave da API da OpenAI ao arquivo `.env`:
     ```
     OPENAI_API_KEY='sua_chave_aqui'
     ```

## Uso

1. **Inicie o servidor backend:**
   ```bash
   uvicorn main_app:app --reload
   ```
   O servidor estará rodando em `http://127.0.0.1:8000`.

2. **Acesse o frontend:**
   - Abra o arquivo `index.html` diretamente no seu navegador.
   - A página irá se comunicar com o backend que está rodando localmente.

## Estrutura da API

O backend expõe vários endpoints, incluindo:

- `GET /`: Serve a página `index.html`.
- `POST /api/ask`: Envia uma pergunta para o chatbot.
- `POST /api/ask_web`: Cria um documento a partir de uma pergunta (simulando busca na web).
- `POST /api/add_to_kb`: Adiciona um documento à base de conhecimento.
- `POST /api/standardize`: Padroniza um documento `.docx` enviado.
- `POST /api/enhance`: Sugere melhorias para um texto de procedimento.
- `POST /api/apply_enhancement`: Aplica as melhorias sugeridas a um documento.
- `POST /api/create`: Cria um novo documento a partir de dados de formulário.