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
    # Filtramos preferentemente modelos de lenguaje/chat
    modelos = [
        m.id
        for m in response.data
        if "whisper" not in m.id and "guard" not in m.id
    ]
    return sorted(modelos) if modelos else ["llama-3.3-70b-versatile"]
  except Exception:
    return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


modelo_embeddings = cargar_modelo_embeddings()

documentos = [
    (
        "Las devoluciones se aceptan hasta 30 días después de la compra, con"
        " el producto en su empaque original."
    ),
    (
        "Los envíos internacionales no tienen devolución gratuita; el cliente"
        " cubre el costo de envío de regreso."
    ),
    (
        "Los productos en oferta o liquidación no son elegibles para"
        " devolución, solo para cambio de talla."
    ),
]

embeddings_documentos = modelo_embeddings.encode(documentos)


def buscar_fragmento(pregunta):
  embedding_pregunta = modelo_embeddings.encode([pregunta])
  similitudes = np.dot(embeddings_documentos, embedding_pregunta.T).flatten()
  return documentos[np.argmax(similitudes)]


st.title("Asistente de Soporte RAG")

lista_modelos = obtener_modelos_groq()
modelo_seleccionado = st.selectbox(
    "Selecciona el modelo de Groq:", lista_modelos
)

pregunta = st.text_input("¿Qué duda tienes sobre nuestras políticas?")

if pregunta:
  fragmento = buscar_fragmento(pregunta)
  prompt_rag = f"""Responde la pregunta del cliente usando SOLO la siguiente política de la tienda. Si la política no cubre la pregunta, dilo claramente.

Política: {fragmento}
Pregunta: {pregunta}
Respuesta muy breve y corta."""

  try:
    response = client.chat.completions.create(
        model=modelo_seleccionado, messages=[{"role": "user", "content": prompt_rag}]
    )
    st.write("**Respuesta:**", response.choices[0].message.content)
  except Exception as e:
    st.error(f"Error al conectar con Groq: {e}")