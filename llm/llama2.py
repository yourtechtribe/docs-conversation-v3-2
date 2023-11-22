# llm.py

import os
import openai
import re
import together
import textwrap
import requests
import asyncio
import logging
import langchain

from requests.exceptions import HTTPError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# Revisar esto de HuggingFAce
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Pinecone

from langchain import PromptTemplate, LLMChain
from typing import Any, Dict, List, Mapping, Optional
from pydantic import BaseModel, root_validator 

# from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

from dotenv import load_dotenv


import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from langchain.vectorstores import Chroma
from llama_index.storage.storage_context import StorageContext

from langchain.vectorstores import Weaviate

from langchain.embeddings import OpenAIEmbeddings


from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.pydantic_v1 import BaseModel

from langchain.chains import RetrievalQA

from IPython.display import Markdown, display

import pandas as pd
from llama_index.query_engine import PandasQueryEngine

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

# https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.weaviate.Weaviate.html

# text clean up

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(lines)

    return wrapped_text

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])


def init(prompt, db):
    """loader = DirectoryLoader('../assets/docs', glob="./*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
    docs = text_splitter.split_documents(documents)

    model_name = "BAAI/bge-base-en"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )

    pinecone.init(
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_ENV
    )

    index_name = "source-embed"

    # If you already have an index, you can load it like this
    text_embed = Pinecone.from_existing_index(index_name, embeddings)"""

    
    # get collection
    # chroma_collection = db.get_or_create_collection("dataset")

    # assign chroma as the vector_store to the context
    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    # index = VectorStoreIndex.from_vector_store(
    #     vector_store, storage_context=storage_context
    # )

    # https://github.com/langchain-ai/langchain/blob/master/templates/rag-weaviate/rag_weaviate/chain.py

    tog_llm = TogetherLLM(
        model= "togethercomputer/llama-2-70b-chat",
        temperature=0.1,
        max_tokens=512
    )

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = ""
    
    # RAG prompt
    sys_prompt = "Eres un asistente virtual capaz de responder a las preguntas (Q&A) de diferentes archivos"
    instruction = "Context: {context}. Responde al usuario siempre en Español al prompt: {question}"
    
    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template
    
    get_prompt(instruction, sys_prompt)

    my_template = get_prompt(instruction, sys_prompt)

    llama_prompt = PromptTemplate(
    template=my_template, input_variables=["context", "question"])


    chain_type_kwargs = {"prompt":llama_prompt}

    # initialize client
    # https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb

    # vectordb = Chroma(persist_directory="./chroma_db") # initialize client
    # Now we can load the persisted database from disk, and use it as normal. 

    qa = RetrievalQA.from_chain_type(llm=tog_llm, 
                                    chain_type="stuff", 
                                    retriever=db.as_retriever(search_kwargs={"k": 3}),
                                    chain_type_kwargs=chain_type_kwargs,
                                    return_source_documents=True)

    llm_response = qa(prompt)
    return process_llm_response(llm_response)

def init_csv(prompt, db):
    response = db.query(prompt)
    return response.metadata["pandas_instruction_str"]

together.api_key = os.environ["TOGETHER_API_KEY"]
together.Models.start("togethercomputer/llama-2-70b-chat")

class TogetherLLM(LLM):
    """Together large language models."""

    model = "togethercomputer/llama-2-70b-chat"
    together_api_key = os.environ["TOGETHER_API_KEY"]
    temperature = 0.7
    max_tokens = 512

    class Config:
        extra = 'forbid'  # Actualizado aquí

    # Está instalada la versión 1 de pydantic sino este decorador da error
    # @root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self):
        return "together"

    def _call(
        self,
        prompt,
        **kwargs: Any,
    ):
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text


"""""
try:
    query = "Que tal?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
except HTTPError as e:
    print(f"Ocurrió un error: {e}")
    print("Response content:", e.response.content.decode())
"""
