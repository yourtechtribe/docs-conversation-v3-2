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

# Interfaz principal
def main():
    st.title('❤ Fertypharm')

    st.markdown("""
        # Bienvenid@ a tu intranet

        Esta aplicación te permite cargar y analizar tus documentos. Aquí puedes:

        - **Cargar documentos**: Sube archivos en formatos PDF, CSV o TXT.
        - **Interactuar con tus documentos**: Una vez cargados, puedes interactuar con tus documentos, realizar preguntas y obtener respuestas basadas en el contenido de los documentos.
        - **Analizar datos**: Para archivos CSV, puedes realizar análisis de datos y visualizaciones.

        ## ¿Cómo usar la aplicación?

        1. Selecciona el tipo de archivo que deseas cargar.
        2. Una vez cargado el archivo, utiliza la interfaz de chat para hacer preguntas sobre el contenido del archivo.
        3. Para archivos CSV, explora las opciones de análisis de datos disponibles.

        ¡Comienza cargando un archivo y descubre lo que nuestra aplicación puede hacer por ti!
        """)

if __name__ == "__main__":
    main()