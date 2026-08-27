import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource
def cargar_modelo_embeddings():
  return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def obtener_modelos_groq():
  try:
    response = client.models.list()
    modelos = [
        m.id
        for m in response.data
        if "whisper" not in m.id and "guard" not in m.id
    ]
    return sorted(modelos) if modelos else ["llama-3.3-70b-versatile"]
  except Exception:
    return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


with st.spinner("Loading..."):
  modelo_embeddings = cargar_modelo_embeddings()
  lista_modelos = obtener_modelos_groq()

st.title("Industrial Technical RAG Assistant")

modelo_seleccionado = st.selectbox(
    "Select the Groq model:", lista_modelos
)

archivo_pdf = st.file_uploader(
    "Upload technical manual or PDF document", type=["pdf"]
)

documentos = []
if archivo_pdf is not None:
  lector = PdfReader(archivo_pdf)
  texto_completo = ""
  for pagina in lector.pages:
    texto = pagina.extract_text()
    if texto:
      texto_completo += texto + "\n"

  documentos = [
      p.strip() for p in texto_completo.split("\n\n") if len(p.strip()) > 30
  ]

  if documentos:
    st.success(
        f"PDF successfully loaded and tokenized. Extracted {len(documentos)}"
        " chunks."
    )
  else:
    st.warning("Could not extract enough text from the PDF.")

if documentos:
  embeddings_documentos = modelo_embeddings.encode(documentos)


  def buscar_fragmento(pregunta):
    embedding_pregunta = modelo_embeddings.encode([pregunta])
    similitudes = np.dot(embeddings_documentos, embedding_pregunta.T).flatten()
    return documentos[np.argmax(similitudes)]


  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if prompt := st.chat_input("Ask a technical question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    fragmento = buscar_fragmento(prompt)
    prompt_rag = f"""Answer the technical question using ONLY the following document excerpt. If the information does not cover the question, state it clearly.

Document Excerpt: {fragmento}
Question: {prompt}
Very short and concise answer."""

    try:
      with st.spinner("Loading..."):
        response = client.chat.completions.create(
            model=modelo_seleccionado,
            messages=[{"role": "user", "content": prompt_rag}],
        )
        respuesta = response.choices[0].message.content
    except Exception as e:
      respuesta = f"Error connecting to Groq: {e}"

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
      st.markdown(respuesta)
else:
  st.info("Please upload a PDF technical manual to start the assistant.")