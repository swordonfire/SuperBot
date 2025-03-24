# SuperBot 🤖 - AI Communication Assistant 
An AI-powered assistant to streamline knowledge management, member discovery, and content generation across Telegram and Twitter, while ensuring privacy with local LLM deployment.

This project implements an AI-powered communication assistant, designed to streamline content creation, management, and community interaction across Telegram and Twitter.  It leverages local Large Language Models (LLMs) for enhanced data privacy and efficient operation, even on resource-constrained devices. This MVP provides a foundation for a comprehensive AI system to empower organizations.

## Screenshots

This screenshot shows the main interface of the Telegram bot, where users can query the knowledge base.
![Screenshot Telegram main interface](/static/images/TelegramBot.png)
 

<br></br>
This screenshot shows the available commands in the  main interface of the Telegram bot.
![Screenshot Telegram Commands main interface](/static/images/TelegramBotCommands.png)


## Features

This MVP includes the following key features:

1.  **Telegram Knowledge Portal Bot:** 🧠

    *   A Telegram bot acting as a central knowledge base for the organization.
    *   Admin UI for easy document uploads to train the bot.
    *   Accurate responses, avoiding hallucinations and confidently stating "NO" when unable to provide an answer.
    *   Implemented using Retrieval-Augmented Generation (RAG) for improved accuracy and context awareness.

2.  **Member Finder:** 🤝

    *   AI-powered matching of community needs with organization members based on a JSON database.
    *   Example query: "I want to find a RUST developer to build a DEFI project with Twitter integration."
    *   Returns the most relevant member(s) with explanations or "NO" if no match is found.

3.  **Twitter Management Assistant:** 🐦

    *   Integration with organization's Twitter account.
    *   AI-driven tweet suggestions for human approval.
    *   Assistance with tweet draft iteration:
        *   Keyword suggestions.
        *   Automatic correction of Twitter handles from the organization's followed accounts list.
    *   Facilitates finalizing and publishing tweets.

4.  **Content Advisor for Telegram and Twitter:** ✨

    *   AI assistance for creating and refining content for both Telegram and Twitter.
    *   Collaborative message refinement with human admins before posting.

5.  **Local LLM Deployment:** 🔒

    *   Runs with local LLM models (Llama.cpp) to ensure data privacy and security.  Leverages GGUF format for efficient inference.

## Technical Solution

*   **LLM Inference:** Llama.cpp is used for local LLM inference, supporting GGUF format models for resource efficiency. Future plans include using `vllm` for distributed inference across multiple GPUs.
*   **Embeddings:** The `avsolatorio/GIST-small-Embedding-v0` embedding model is used for generating text embeddings, optimized for devices with limited GPU resources.
*   **Vector Database:** ChromaDB is employed as the vector store for efficient retrieval of relevant information.  Future plans include migrating to Weaviate for advanced features like hybrid search.
*   **Telegram Bot:** `python-telegram-bot` library facilitates the creation and management of the Telegram bot.
*   **Web Framework:** FastAPI and Pydantic are used for building the API with type safety and performance in mind.
*   **Development Tools:** `uv` is used for environment management and dependency installation, offering a faster alternative to `pip`. Dockerization is planned for future deployments.
*   **RAG Implementation:** Langchain is planned for integration to provide a robust and flexible RAG framework.

## Getting Started

### Prerequisites

*   Python 3.9+
*   `uv` (recommended for faster dependency management) or `pip`
*   Access to a Telegram bot token
*   Organization's member data in JSON format

### Installation

1.  **Clone the repository:**

```bash
git clone https://github.com/swordonfire/SuperBot.git
cd SuperBot
```
2. **Create and activate a virtual environment (using uv - recommended):**

```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Or using **venv**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. **Install dependencies (using uv - recommended):**
```bash
uv add -r requirements.txt
```
Or using **pip**:
```bash
pip install -r requirements.txt
```
4. **Download the LLM model:**
```bash
python scripts/download_models.py --repo_id lmstudio-community/Llama-3.2-3B-Instruct-GGUF --model_file Llama-3.2-3B-Instruct-Q8_0.gguf
```
5. **Configure environment variables:**
Edit .env.example (save it as .env file) and set the TELEGRAM_BOT_TOKEN, LLM_MODEL_PATH, EMBEDDING_MODEL, and PROJECT_NAME parameters.  Get your Telegram bot token from BotFather.
6. **Prepare the member data:**
Ensure the member data is in a JSON file (e.g., members.json) with the required format.  The structure should be suitable for querying based on skills and other criteria.  Place this file in the data directory.
<br></br>

**Running the Telegram Bot**
```bash
PYTHONPATH=. python -m src.services.telegram_service
```
**Part 4: Directory Structure**
```bash
SuperBot/
├── data/                  # Document and model storage
├── src/                   # Application code
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Configuration and utilities
│   ├── llm/               # Local LLM integration
│   ├── rag/               # RAG pipeline components
│   └── services/          # Telegram and Twitter services
├── tests/                 # Unit and integration tests
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

**Part 5: Future Enhancements, Contributing, and License**

## Future Enhancements

*   Integration with Langchain for enhanced RAG functionality.
*   Migration to Weaviate for advanced vector search capabilities.
*   Dockerization for easier deployment and portability.
*   Expanding the Twitter management features.
*   Implementing a user-friendly web interface for admins.
*   Further optimization for resource-constrained environments.

## Contributing

Contributions are welcome\! Please open an issue or submit a pull request.

## License

MIT License (See [LICENSE.md](https://github.com/swordonfire/SuperBot/blob/main/LICENSE))
