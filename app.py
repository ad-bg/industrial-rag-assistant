import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
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

documentos = [
    (
        "Returns are accepted up to 30 days after purchase, with the product in"
        " its original packaging."
    ),
    (
        "International shipping does not have free returns; the customer covers"
        " the return shipping cost."
    ),
    (
        "Sale or clearance products are not eligible for return, only for size"
        " exchange."
    ),
]

embeddings_documentos = modelo_embeddings.encode(documentos)


def buscar_fragmento(pregunta):
  embedding_pregunta = modelo_embeddings.encode([pregunta])
  similitudes = np.dot(embeddings_documentos, embedding_pregunta.T).flatten()
  return documentos[np.argmax(similitudes)]


st.title("RAG Support Assistant")

modelo_seleccionado = st.selectbox(
    "Select the Groq model:", lista_modelos
)

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("Type your question about the policies..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  fragmento = buscar_fragmento(prompt)
  prompt_rag = f"""Answer the customer's question using ONLY the following store policy. If the policy does not cover the question, state it clearly.

Policy: {fragmento}
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