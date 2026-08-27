# Industrial Technical RAG Assistant

## Overview

Industrial Technical RAG Assistant is a web application for querying technical
manuals through a chat powered by **Retrieval-Augmented Generation (RAG)**.
Users upload a PDF document, ask questions in natural language, and receive
concise answers based on the most relevant document excerpt.

The project is designed for industrial documentation, maintenance manuals,
technical datasheets, and automation material. Semantic retrieval and the
embedding model run locally, while answer generation is handled through the
Groq API.

### Key features

- Upload PDF documents through a Streamlit interface.
- Extract text page by page with PyPDF.
- Split document content into retrieval-ready chunks.
- Generate multilingual embeddings with
  `paraphrase-multilingual-MiniLM-L12-v2`.
- Dynamically list available Groq models, including Llama models.
- Maintain conversational history during the session.
- Use an RAG prompt that requires the model to answer only from the retrieved
  context and clearly state when the document does not contain the answer.

## Technologies

| Technology | Purpose |
| --- | --- |
| Python 3.10+ | Core programming language |
| Streamlit | Web interface and chat experience |
| Groq API | Language model inference, including Llama 3 |
| Sentence Transformers | Semantic embedding generation |
| NumPy | Vector similarity calculations |
| PyPDF | PDF reading and text extraction |
| python-dotenv | Environment variable loading |
| PyTorch | Runtime required by Sentence Transformers |

## Prerequisites

- Python 3.10 or later.
- A Groq API key, available from
  [console.groq.com](https://console.groq.com/).
- An Internet connection for Groq requests and for downloading the embedding
  model on its first run.

## Installation

### 1. Clone or download the project

Open a terminal in the project directory:

```powershell
cd "C:\ruta\al\proyecto\RAG"
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents script activation, enable script execution for your user
or activate the environment from `cmd.exe`:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root and add:

```dotenv
GROQ_API_KEY=tu_clave_de_groq
```

Do not share this file or commit the key to a repository. Add `.env` to
`.gitignore` before publishing the project.

### 5. Run the application

```powershell
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`, which you
can open in your browser.

## How the system works

The RAG pipeline runs through the following steps:

1. **Document upload:** the user uploads a `.pdf` file through the interface.
2. **Extraction and preparation:** `PdfReader` iterates over the pages,
  extracts available text, and splits it into paragraph-based chunks.
3. **Semantic indexing:** Sentence Transformers converts each chunk into a
  numerical vector. The model is cached to avoid repeated loading.
4. **Retrieval:** when a question is submitted, the application generates its
  embedding and calculates dot-product similarity against the document
  embeddings. The highest-scoring chunk is selected.
5. **Controlled generation:** the retrieved chunk and question are sent to
  Groq. The prompt instructs the model to use only that chunk and explicitly
  acknowledge missing information.
6. **Response:** the answer is displayed in the chat and retained in the
  Streamlit session history.

## Project structure

```text
RAG/
├── app.py             # Streamlit application and RAG pipeline
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .env               # Groq key; do not commit
```

## Considerations and limitations

- The PDF must contain selectable text. Scanned documents without a text layer
  require OCR before they can be used.
- Each query retrieves one chunk, so an answer that depends on information
  spread across multiple sections may be incomplete.
- The question and retrieved excerpt are sent to Groq to generate the answer.
  Do not upload confidential information without reviewing the applicable
  policies.
- Answer quality depends on PDF extraction quality, chunking, and the selected
  model.

## License

This project is licensed under the [MIT License](LICENSE).