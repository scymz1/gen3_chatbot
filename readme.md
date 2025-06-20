
# gen3-helm-uf-cancer-data-center

This is uf cancer-data-center application development code using gen3. To start helm deployment:

## 🚀 Chatbot Service

This is an intelligent chatbot backend service based on FastAPI and Llama-3, supporting multi-session, history recording, and user authentication.

### ✨ Key Features

- **Intelligent Chat**: Integrates the Llama-3 8B model for a smooth conversational experience.
- **Multi-Session Management**: Supports creating, switching, and renaming multiple user sessions.
- **History Recording**: Automatically saves all conversation history for easy review.
- **CORS Support**: Configured for cross-origin requests, facilitating separate frontend/backend development.
- **Dockerized**: Includes a Dockerfile for one-click build and deployment.

### 快速开始 (Quick Start)

#### 1. Prerequisites

- [Docker](https://www.docker.com/)
- [Python 3.10+](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)
- [uvicorn](https://www.uvicorn.org/)

#### 2. Local Development

1. **Clone the Project**
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model**
   Please download the `Meta-Llama-3-8B-Instruct.Q2_K.gguf` model from Hugging Face or other sources and place it in the `./model` directory.

4. **Start the Service**
   ```bash
   uvicorn backend:app --reload --host 0.0.0.0 --port 8000
   ```
   Once the service is running, you can access the API documentation at http://localhost:8000/docs.

#### 3. Docker Deployment

1. **Build the Docker Image**
   ```bash
   docker build -t chatbot .
   ```

2. **Run the Docker Container**
   ```bash
   docker run -it --rm -p 8000:8000 chatbot
   ```

3. **Push to Docker Hub**
   ```bash
   docker tag chatbot minghaozhou01/chatbot:latest
   docker push minghaozhou01/chatbot:latest
   ```

### 📦 API Endpoints

- `POST /chatbot/new_conversation`: Create a new conversation
- `GET /chatbot/conversations`: Get the list of conversations
- `GET /chatbot/history`: Get the history of a conversation
- `POST /chatbot/rename_conversation`: Rename a conversation
- `POST /chatbot/chat`: Send a message

### ⚙️ Customization

#### Accessing Guppy Data

To allow the LLM to access Guppy data, execute the following in your Kubernetes environment:

```bash
kubectl exec -it <your-pod-name> -- bash
python fetch_guppy_data.py
```

---

other codes I used myself: