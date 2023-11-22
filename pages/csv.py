import streamlit as st
import pandas as pd
import pdfplumber
import tempfile
import uuid
import shutil
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import together
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import pandas as pd
from io import StringIO

import os
import time

import pandas as pd
from llama_index.query_engine import PandasQueryEngine

from llm import llama2

# Configuración inicial de Streamlit
st.set_page_config(page_title='Your Tech Tribe <> Fertypharm', page_icon="🛖", layout="wide")  

# Función para guardar el archivo cargado en una carpeta local
def guardar_archivo_local(archivo_cargado):
    if archivo_cargado is not None:
        # Crear una carpeta 'archivos_subidos' si no existe
        if not os.path.exists('docs'):
            os.makedirs('docs')

        # Crear la ruta del archivo donde se guardará
        archivo_ruta = os.path.join('docs', archivo_cargado.name)

        # Escribir el archivo cargado en el sistema de archivos
        with open(archivo_ruta, "wb") as f:
            f.write(archivo_cargado.getbuffer())
        
        return archivo_ruta
    return None

# Función para procesar diferentes tipos de archivos
def procesar_archivo(archivo):
    
    if archivo.type == "text/csv":
        archivo_ruta = guardar_archivo_local(archivo)

        df = pd.read_csv(archivo_ruta)
        print(df)
        query_engine = PandasQueryEngine(df=df, verbose=True)

        return query_engine, df
    else:
        return None

# Función para generar respuestas utilizando Together y Langchain
def generar_respuesta(contenido, mensaje_usuario):
    # Implementación de la lógica para generar respuestas
    # (Aquí se puede integrar la lógica de tu código actual)
    # Por ahora, retorna un mensaje genérico
    return "Respuesta generada al mensaje: " + mensaje_usuario

#def clear_chat_history():
#    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
#     st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Interfaz principal
def cargar_archivo():
    st.title('🔗 Habla con tu CSV 📄')

    # Carga de archivos
    archivo_cargado = st.file_uploader("Carga tu documento (PDF, CSV, TXT)", type=["pdf", "csv", "txt"])

    if archivo_cargado is not None:
        
        qa, dataframe = procesar_archivo(archivo_cargado)

        if qa:
            st.session_state['contenido'] = qa
            st.session_state['df'] = dataframe
            st.success("El archivo se ha procesado correctamente.")
            main()
        else:
            st.error("Formato de archivo no soportado.")

def main():
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input(disabled= not  st.session_state['contenido']):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("🗨 ..."):
                
                pandas_instruction = llama2.init_csv(prompt, st.session_state['contenido'])
                try:
                    result = eval(pandas_instruction, {'df': st.session_state['df']})
                    placeholder = st.empty()
                    placeholder.markdown(result)
                except Exception as e:
                    st.error(f"Error al procesar la instrucción de pandas: {e}") 
        message = {"role": "assistant", "content": result}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    cargar_archivo()