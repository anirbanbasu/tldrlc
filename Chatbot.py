# pylint: disable=line-too-long
# pylint: disable=too-many-lines
# pylint: disable=invalid-name
# pylint: disable=broad-exception-caught
# pylint: disable=use-yield-from

# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Module for the main page of the Streamlit app for TLDRLC."""

import datetime
import hashlib
import json
import os
import time
import types
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from pyvis.network import Network
import networkx as nx
import wikipedia
from llama_index.core.callbacks import CallbackManager
from llama_index.core import KnowledgeGraphIndex
from llama_index.readers.web import BeautifulSoupWebReader, TrafilaturaWebReader
from llama_index.readers.papers import ArxivReader, PubmedReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.cohere_rerank import CohereRerank

import streamlit as st

# from langfuse.llama_index import LlamaIndexCallbackHandler
from langfuse import Langfuse
from utils.callbacks.ragas_langfuse import (
    RagasLangfuseCallbackHandler,
)


# Fun emojis for messages: https://github.com/ikatyang/emoji-cheat-sheet


# Load environment variables from .env file, if present
load_dotenv()

# Setup page metadata
st.set_page_config(
    page_title="TLDRLC: Too Long, Didn't Read, Let's Chat!",
    layout="wide",
    menu_items={
        "About": """
        An experimental chatbot that lets the user chat about document sources 
        using a Knowledge Graph (KG) based Retrieval Augmented Generation (RAG). 
        Check out the project on [Github](https://github.com/anirbanbasu/tldrlc).
        """,
        "Report a Bug": "https://github.com/anirbanbasu/tldrlc/issues",
    },
    initial_sidebar_state="expanded",
    page_icon="🤖",
)

# Constants
# Default Wikipedia language prefix
const_default_wikipedia_language_prefix: str = "en"

# Mechanism to ensure that certain things run only once
if "first_run" not in st.session_state.keys():
    st.session_state.first_run = True

# Setup performance evaluation using Langfuse
if "use_langfuse" not in st.session_state.keys():
    st.session_state.use_langfuse = bool(
        os.getenv("EVAL_USE_LANGFUSE", "True").lower()
        in ["true", "yes", "t", "y", "on"]
    )


def setup_langfuse():
    """Setup (or, disable) Langfuse for performance evaluation."""
    if st.session_state.use_langfuse:

        langfuse__secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse__public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse__host = os.getenv("LANGFUSE_HOST")
        st.session_state.langfuse = Langfuse(
            public_key=langfuse__public_key,
            secret_key=langfuse__secret_key,
            host=langfuse__host,
        )
        try:
            st.session_state.langfuse.auth_check()
            # Setup LangFuse tags for the handler, if necessary
            env__langfuse_trace_tags = os.getenv("LANGFUSE_TRACE_TAGS", None)
            langfuse_trace_tags = None
            if env__langfuse_trace_tags is not None:
                langfuse_trace_tags = [
                    x.strip() for x in env__langfuse_trace_tags.split(",")
                ]
            # Setup the callback handler
            st.session_state.langfuse_callback_handler = RagasLangfuseCallbackHandler(
                secret_key=langfuse__secret_key,
                public_key=langfuse__public_key,
                host=langfuse__host,
                tags=langfuse_trace_tags,
            )
            Settings.callback_manager = CallbackManager(
                [st.session_state.langfuse_callback_handler]
            )
            st.toast(
                f":white_check_mark: Using Langfuse at {os.getenv('LANGFUSE_HOST')} for performance evaluation."
            )
        except Exception as langfuse_e:
            st.error(
                f":x: {langfuse_e} Langfuse authentication failed. Disabling Langfuse."
            )
            st.session_state.use_langfuse = False
            st.session_state.langfuse_callback_handler = None
    else:
        st.session_state.langfuse_callback_handler = None
        st.toast(":warning: Not using Langfuse for performance evaluation.")


if st.session_state.first_run:
    setup_langfuse()

# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Supported LLM providers
const_llm_providers = types.SimpleNamespace()
const_llm_providers.list = ["Cohere", "Ollama", "Open AI"]
const_llm_providers.cohere = const_llm_providers.list[0]
const_llm_providers.ollama = const_llm_providers.list[1]
const_llm_providers.openai = const_llm_providers.list[2]

# LLM settings
if "settings_llm__provider" not in st.session_state.keys():
    st.session_state.settings_llm__provider = os.getenv("LLM_PROVIDER", "Ollama")

if "settings_llm__request_timeout" not in st.session_state.keys():
    st.session_state.settings_llm__request_timeout = int(
        os.getenv("LLM_REQUEST_TIMEOUT", "120")
    )
if "settings_llm__llm_temperature" not in st.session_state.keys():
    st.session_state.settings_llm__llm_temperature = float(
        os.getenv("LLM_TEMPERATURE", "0.0")
    )
if "settings_llm__llm_system_message" not in st.session_state.keys():
    st.session_state.settings_llm__llm_system_message = os.getenv(
        "LLM_SYSTEM_MESSAGE",
        "You are an intelligent assistant. You respond to questions about articles from various sources such as Wikipedia, arXiv, Pubmed and so on. You generate your answers based ONLY on information in those articles that are provided to you as context information. If you are unsure about an answer or if the user query cannot be answered using information in the context then say that you do not know. If the user ask you to stop, immediately stop answering.",
    )
if "settings_llm__llm_chunk_size" not in st.session_state.keys():
    st.session_state.settings_llm__llm_chunk_size = int(
        os.getenv("LLM_CHUNK_SIZE", "1024")
    )
if "settings_llm__llm_chunk_overlap" not in st.session_state.keys():
    st.session_state.settings_llm__llm_chunk_overlap = int(
        os.getenv("LLM_CHUNK_OVERLAP", "64")
    )
if "settings_llm__index_memory_token_limit" not in st.session_state.keys():
    st.session_state.settings_llm__index_memory_token_limit = int(
        os.getenv("INDEX_MEMORY_TOKEN_LIMIT", "4096")
    )
if "settings_llm__index_max_triplets_per_chunk" not in st.session_state.keys():
    st.session_state.settings_llm__index_max_triplets_per_chunk = int(
        os.getenv("INDEX_MAX_TRIPLETS_PER_CHUNK", "8")
    )
if "settings_llm__index_include_embeddings" not in st.session_state.keys():
    st.session_state.settings_llm__index_include_embeddings = bool(
        os.getenv("INDEX_INCLUDE_EMBEDDINGS", "True").lower()
        in ["true", "yes", "t", "y", "on"]
    )
if "settings_llm__index_chat_mode" not in st.session_state.keys():
    st.session_state.settings_llm__index_chat_mode = os.getenv(
        "INDEX_CHAT_MODE", "context"
    )

# Graph database settings
if "settings_graphdb__disable" not in st.session_state.keys():
    st.session_state.settings_graphdb__disable = bool(
        os.getenv("NEO4J_DISABLE", "False").lower() in ["true", "yes", "t", "y", "on"]
    )
if "settings_graphdb__url" not in st.session_state.keys():
    st.session_state.settings_graphdb__url = os.getenv(
        "NEO4J_URL", "bolt://localhost:7687"
    )
if "settings_graphdb__username" not in st.session_state.keys():
    st.session_state.settings_graphdb__username = os.getenv("NEO4J_USERNAME", "neo4j")
if "settings_graphdb__password" not in st.session_state.keys():
    st.session_state.settings_graphdb__password = os.getenv("NEO4J_PASSWORD")
if "settings_graphdb__dbname" not in st.session_state.keys():
    st.session_state.settings_graphdb__dbname = os.getenv("NEO4J_DB_NAME", "neo4j")

# Redis settings
if "settings_redis__disable" not in st.session_state.keys():
    st.session_state.settings_redis__disable = bool(
        os.getenv("REDIS_DISABLE", "False").lower() in ["true", "yes", "t", "y", "on"]
    )
if "settings_redis__url" not in st.session_state.keys():
    st.session_state.settings_redis__url = os.getenv(
        "REDIS_URL", "redis://localhost:6379"
    )
if "settings_redis__namespace" not in st.session_state.keys():
    st.session_state.settings_redis__namespace = os.getenv("REDIS_NAMESPACE", "tldrlc")

# Knowledge graph visualisation settings
if "settings_kgvis__height" not in st.session_state.keys():
    st.session_state.settings_kgvis__height = int(os.getenv("KG_VIS_HEIGHT", "800"))
if "settings_kgvis__filename" not in st.session_state.keys():
    st.session_state.settings_kgvis__filename = os.getenv(
        "KG_VIS_FILENAME", "__tmp_kg.html"
    )
if "settings_kgvis__max_nodes" not in st.session_state.keys():
    st.session_state.settings_kgvis__max_nodes = int(
        os.getenv("KG_VIS_MAX_NODES", "100")
    )
if "settings_kgvis__max_depth" not in st.session_state.keys():
    st.session_state.settings_kgvis__max_depth = int(os.getenv("KG_VIS_MAX_DEPTH", "3"))
if "settings_kgvis__layout" not in st.session_state.keys():
    st.session_state.settings_kgvis__layout = os.getenv("KG_VIS_LAYOUT", "spring")
if "settings_kgvis__physics_enabled" not in st.session_state.keys():
    st.session_state.settings_kgvis__physics_enabled = bool(
        os.getenv("KG_VIS_PHYSICS_ENABLED", "True").lower()
        in ["true", "yes", "t", "y", "on"]
    )

# User interface backed-elements
if "ui__chk_source_rebuild_index" not in st.session_state.keys():
    st.session_state.ui__chk_source_rebuild_index = False

if "ui__txtinput_document_source" not in st.session_state.keys():
    if "index" not in st.session_state.keys():
        st.session_state.ui__txtinput_document_source = os.getenv(
            "DEFAULT_SOURCE", None
        )
    else:
        st.session_state.ui__txtinput_document_source = st.session_state.index.index_id

if "ui__radio_source_type" not in st.session_state.keys():
    st.session_state.ui__radio_source_type = "Wikipedia"

if "ui__selectbox_wikipedia_prefix" not in st.session_state.keys():
    st.session_state.ui__selectbox_wikipedia_prefix = (
        const_default_wikipedia_language_prefix
    )

if "ui__select_webpage_reader" not in st.session_state.keys():
    st.session_state.ui__select_webpage_reader = "BeautifulSoup"


def require_force_index_rebuild():
    """Force the index to be rebuilt when the source document(s) have changed."""
    st.session_state.ui__chk_source_rebuild_index = True
    st.cache_resource.clear()


def update_openai_apikey():
    """Update the OpenAI API key in the OS environment variable."""
    os.environ["OPENAI_API_KEY"] = st.session_state.settings_llm__openai_api_key
    update_llm_settings()


def update_cohere_apikey():
    """Update the Cohere API key in the OS environment variable."""
    os.environ["COHERE_API_KEY"] = st.session_state.settings_llm__cohere_api_key
    update_llm_settings()


def update_llm_settings():
    """Update the language model settings."""
    try:
        match st.session_state.settings_llm__provider:
            case const_llm_providers.cohere:
                if "settings_llm__cohere_api_key" not in st.session_state.keys():
                    st.session_state.settings_llm__cohere_api_key = os.getenv(
                        "COHERE_API_KEY"
                    )
                st.session_state.settings_llm__model = os.getenv(
                    "COHERE_MODEL", "command"
                )
                st.session_state.client = Cohere(
                    api_key=st.session_state.settings_llm__cohere_api_key,
                    model=st.session_state.settings_llm__model,
                    temperature=st.session_state.settings_llm__llm_temperature,
                )
                st.session_state.embed_model = embed_model = CohereEmbedding(
                    cohere_api_key=st.session_state.settings_llm__cohere_api_key,
                    input_type="search_query",
                    # embedding_type="binary",
                )
            case const_llm_providers.openai:
                if "settings_llm__openai_api_key" not in st.session_state.keys():
                    st.session_state.settings_llm__openai_api_key = os.getenv(
                        "OPENAI_API_KEY"
                    )
                st.session_state.settings_llm__model = os.getenv(
                    "OPENAI_MODEL", "gpt-3.5-turbo"
                )
                st.session_state.client = OpenAI(
                    model=st.session_state.settings_llm__model,
                    temperature=st.session_state.settings_llm__llm_temperature,
                )
                st.session_state.embed_model = OpenAIEmbedding()
            case const_llm_providers.ollama:
                if "settings_llm__ollama_url" not in st.session_state.keys():
                    st.session_state.settings_llm__ollama_url = os.getenv(
                        "OLLAMA_URL", "http://localhost:11434"
                    )
                st.session_state.settings_llm__model = os.getenv(
                    "OLLAMA_MODEL", "mistral"
                )
                st.session_state.client = Ollama(
                    model=st.session_state.settings_llm__model,
                    request_timeout=st.session_state.settings_llm__request_timeout,
                    base_url=st.session_state.settings_llm__ollama_url,
                    temperature=st.session_state.settings_llm__llm_temperature,
                )
                embed_model = OllamaEmbedding(
                    model_name=st.session_state.settings_llm__model,
                    base_url=st.session_state.settings_llm__ollama_url,
                )
                st.session_state.embed_model = embed_model

        # Global LlamaIndex settings
        Settings.llm = st.session_state.client
        Settings.embed_model = st.session_state.embed_model
        Settings.chunk_size = st.session_state.settings_llm__llm_chunk_size
        Settings.chunk_overlap = st.session_state.settings_llm__llm_chunk_overlap
        st.session_state.chat_store = SimpleChatStore()
        st.session_state.memory = ChatMemoryBuffer.from_defaults(
            token_limit=st.session_state.settings_llm__index_memory_token_limit,
            chat_store=st.session_state.chat_store,
        )
        if not st.session_state.first_run:
            require_force_index_rebuild()
        st.toast(
            f":white_check_mark: Using {st.session_state.settings_llm__provider} {st.session_state.settings_llm__model} language model. {'Cohere Reranker enabled.' if st.session_state.settings_llm__provider == const_llm_providers.cohere else ''}"
        )
    except Exception as llm_e:
        st.error(f":x: {type(llm_e).__name__}: {llm_e}")


def update_graphdb_settings():
    """Update the graph database settings."""
    if not st.session_state.settings_graphdb__disable:
        try:
            st.session_state.graph_store = Neo4jGraphStore(
                username=st.session_state.settings_graphdb__username,
                password=st.session_state.settings_graphdb__password,
                url=st.session_state.settings_graphdb__url,
                database=st.session_state.settings_graphdb__dbname,
            )
            if "storage_context" not in st.session_state.keys():
                st.session_state.storage_context = StorageContext.from_defaults(
                    graph_store=st.session_state.graph_store
                )
            else:
                st.session_state.storage_context.graph_store = (
                    st.session_state.graph_store
                )
            st.toast(
                f":white_check_mark: Connected to graph store {st.session_state.settings_graphdb__dbname} at {st.session_state.settings_graphdb__url} as {st.session_state.settings_graphdb__username}."
            )
        except Exception as graph_e:
            st.error(f":x: {type(graph_e).__name__}: {graph_e}")
    else:
        if "storage_context" not in st.session_state.keys():
            st.session_state.storage_context = StorageContext.from_defaults(
                graph_store=SimpleGraphStore()
            )
        else:
            st.session_state.storage_context.graph_store = SimpleGraphStore()
        st.toast(
            ":warning: Neo4j graph database is disabled. Reverting to in-memory storage."
        )


def update_document_index_store_settings():
    """Update the document and index store settings."""
    if not st.session_state.settings_redis__disable:
        try:
            kv_store = RedisKVStore(
                redis_uri=st.session_state.settings_redis__url,
            )
            st.session_state.document_store = RedisDocumentStore(
                redis_kvstore=kv_store,
                namespace=st.session_state.settings_redis__namespace,
            )
            st.session_state.index_store = RedisIndexStore(
                redis_kvstore=kv_store,
                namespace=st.session_state.settings_redis__namespace,
            )
            if "storage_context" not in st.session_state.keys():
                st.session_state.storage_context = StorageContext.from_defaults(
                    docstore=st.session_state.document_store,
                    index_store=st.session_state.index_store,
                )
            else:
                st.session_state.storage_context.docstore = (
                    st.session_state.document_store
                )
                st.session_state.storage_context.index_store = (
                    st.session_state.index_store
                )
            parsed = urlparse(st.session_state.settings_redis__url)
            if parsed.username:
                st.toast(
                    f":white_check_mark: Set document and index store to {st.session_state.settings_redis__namespace} at {parsed.hostname}:{parsed.port} as user: {parsed.username}."
                )
            else:
                st.toast(
                    f":white_check_mark: Set document and index store to {st.session_state.settings_redis__namespace} at {parsed.hostname}:{parsed.port}."
                )
        except Exception as index_e:
            st.error(f":x: {type(index_e).__name__}: {index_e}")
    else:
        if "storage_context" not in st.session_state.keys():
            st.session_state.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
            )
        else:
            st.session_state.storage_context.docstore = SimpleDocumentStore()
            st.session_state.storage_context.index_store = SimpleIndexStore()
        st.toast(
            ":warning: Redis index and document store is disabled. Reverting to in-memory storage."
        )


def update_graph_visualisation_settings():
    """Update the graph visualisation settings."""
    st.session_state.settings_saved_kgvis__height = (
        st.session_state.settings_kgvis__height
    )
    st.session_state.settings_saved_kgvis__filename = (
        st.session_state.settings_kgvis__filename
    )


if st.session_state.first_run:
    match st.session_state.settings_llm__provider:
        case const_llm_providers.cohere:
            st.session_state.settings_llm__cohere_api_key = os.getenv("COHERE_API_KEY")
        case const_llm_providers.openai:
            st.session_state.settings_llm__openai_api_key = os.getenv("OPENAI_API_KEY")
    update_llm_settings()
    update_graphdb_settings()
    update_document_index_store_settings()
    update_graph_visualisation_settings()
    st.session_state.first_run = False

st.title("TLDRLC: Too Long, Didn't Read, Let's Chat!")
st.markdown(
    """
            This is an experimental chatbot that lets you chat about document sources using a Knowledge Graph (KG) 
            based Retrieval Augmented Generation (RAG).

            **EU AI Act [Article 52](https://artificialintelligenceact.eu/article/52/) Transparency notice**: :red[By using this app, you 
            are interacting with an artificial intelligence (AI) system. You are  advised not to  take any of its responses as facts.
            The AI system is not a substitute for professional advice. If you are unsure about any information, please consult a professional in the field.]
            """
)

# Setup the side bar
st.sidebar.title("Settings")

# Setup the language model settings
with st.sidebar.expander("**Language model**", expanded=False):
    st.selectbox(
        "LLM provider",
        options=const_llm_providers.list,
        key="settings_llm__provider",
        help="This setting determines the language model provider. For cloud-based providers, respective API keys are required. For Ollama, the server URL is required.",
        on_change=update_llm_settings,
    )
    match st.session_state.settings_llm__provider:
        case const_llm_providers.cohere:
            st.text_input(
                "Cohere API key",
                type="password",
                key="settings_llm__cohere_api_key",
                help="The API key for Cohere.",
                on_change=update_cohere_apikey,
            )
        case const_llm_providers.openai:
            st.text_input(
                "Open AI API key",
                type="password",
                key="settings_llm__openai_api_key",
                help="The API key for Open AI.",
                on_change=update_openai_apikey,
            )
        case const_llm_providers.ollama:
            st.text_input(
                "Ollama server URL",
                key="settings_llm__ollama_url",
                help="The URL of the Ollama server. Make sure that the Ollama server is listening on this port of the hostname.",
                on_change=update_llm_settings,
            )
    st.text_input(
        "Language model",
        key="settings_llm__model",
        help="The language model to use with the chosen LLM provider. If using Ollama, the model must be available in the Ollama server. Run `ollama list` to see available models. Or, run `ollama pull <model>` to download a model.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=(
            2.0
            if st.session_state.settings_llm__provider == const_llm_providers.openai
            else 1.0
        ),
        step=0.1,
        key="settings_llm__llm_temperature",
        help="The temperature parameter with a value in $[0,1]$ controls the randomness of the output. A value of $0$ is least random while a value of $1$ will make the output more random, hence more creative.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Chunk size",
        min_value=128,
        max_value=4096,
        step=128,
        key="settings_llm__llm_chunk_size",
        help="The chunk size is the maximum number of tokens in each chunk of text. The language model will process the text in chunks of this size.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Chunk overlap",
        min_value=16,
        max_value=128,
        step=16,
        key="settings_llm__llm_chunk_overlap",
        help="The chunk overlap is the number of tokens that overlap between adjacent chunks. This helps the language model to generate coherent text across chunks.",
        on_change=update_llm_settings,
    )
    st.text_area(
        "System message",
        key="settings_llm__llm_system_message",
        help="The system message is an optional initial prompt for the language model. It should be a short sentence or a few words that describe the context of the conversation. The language model will use this prompt to generate a response.",
        on_change=update_llm_settings,
    )

# Setup the language model chat index settings
with st.sidebar.expander("**Language model chat index**", expanded=False):
    st.slider(
        "Memory token limit",
        min_value=2048,
        max_value=16384,
        step=512,
        key="settings_llm__index_memory_token_limit",
        help="The maximum number of tokens to store in the chat memory buffer.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Max triplets per chunk",
        min_value=1,
        max_value=32,
        step=1,
        key="settings_llm__index_max_triplets_per_chunk",
        help="The maximum number of knowledge graph triplets to extract from each chunk of text.",
        on_change=update_llm_settings,
    )
    st.selectbox(
        "Chat mode",
        options=["condense_plus_context", "context"],
        disabled=True,
        key="settings_llm__index_chat_mode",
        help="The chat mode determines how the knowledge graph is used to generate responses. The `condense_plus_context` mode condenses the question in the prompt first and then uses the knowledge graph to provide context. The `context` mode uses the knowledge graph to provide context for the language model but does not condense the input prompt.",
        on_change=update_llm_settings,
    )
    st.checkbox(
        "Include embeddings in index",
        key="settings_llm__index_include_embeddings",
        help="If checked, the knowledge graph index will include embeddings for each document.",
        on_change=update_llm_settings,
    )

# Setup the graph database settings
with st.sidebar.expander("**Graph storage**", expanded=False):
    st.checkbox(
        "Do not use graph database",
        key="settings_graphdb__disable",
        help="If checked, the knowledge graph will reside entirely in memory.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Neo4j URL",
        disabled=st.session_state.settings_graphdb__disable,
        key="settings_graphdb__url",
        help="The URL of the on-premises Neo4j database or the cloud-hosted Neo4j Aura DB.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Username",
        disabled=st.session_state.settings_graphdb__disable,
        key="settings_graphdb__username",
        help="The username to connect to the Neo4j database.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Password",
        disabled=st.session_state.settings_graphdb__disable,
        key="settings_graphdb__password",
        type="password",
        help="The password to connect to the Neo4j database.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Database name",
        disabled=st.session_state.settings_graphdb__disable,
        key="settings_graphdb__dbname",
        help="The name of the Neo4j database.",
        on_change=update_graphdb_settings,
    )
    st.markdown(
        """
        #### :cloud: Remote Neo4j
        If you need a cloud-hosted Neo4j instance, you can try [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/).
        """
    )


# Setup the document and index storage settings
with st.sidebar.expander("**Document and index storage**", expanded=False):
    st.checkbox(
        "Do not use Redis",
        key="settings_redis__disable",
        help="If checked, the index and documents will reside entirely in memory.",
        on_change=update_document_index_store_settings,
    )
    st.text_input(
        label="Redis connection URL",
        disabled=st.session_state.settings_redis__disable,
        key="settings_redis__url",
        help="The URL of the Redis key-value storage.",
        type="password",
        on_change=update_document_index_store_settings,
    )
    st.text_input(
        label="Redis namespace",
        disabled=st.session_state.settings_redis__disable,
        key="settings_redis__namespace",
        help="The namespace of the Redis storage.",
        on_change=update_document_index_store_settings,
    )
    st.markdown(
        """
        #### :cloud: Remote Redis
        If you need a cloud-hosted Redis instance, you can try [Render](https://www.render.com/).
        """
    )


# Setup the graph visualisation settings
with st.sidebar.expander("**Graph visualisation**", expanded=False):
    st.text_input(
        label="Visualisation file",
        key="settings_kgvis__filename",
        help="The name of the file where the visualisation is saved.",
        on_change=update_graph_visualisation_settings,
    )
    st.slider(
        "Max nodes",
        min_value=25,
        max_value=100,
        step=1,
        key="settings_kgvis__max_nodes",
        help="The maximum number of nodes, starting with the most connected ones, to display in the visualisation.",
    )
    st.slider(
        "Height (in pixels)",
        min_value=250,
        max_value=2500,
        step=50,
        key="settings_kgvis__height",
        help="The height, in pixels, of the knowledge graph rendering.",
        on_change=update_graph_visualisation_settings,
    )
    st.selectbox(
        "Graph layout",
        options=["circular", "planar", "shell", "spectral", "spring"],
        key="settings_kgvis__layout",
        help="The visual layout mode of the knowledge graph.",
    )
    st.checkbox(
        "Physics enabled",
        key="settings_kgvis__physics_enabled",
        help="If checked, the physics simulation will be enabled for the knowledge graph rendering.",
    )


def stream_wrapper(streaming_response):
    """Wrapper for the streaming response from the chat engine."""
    # Do not use yield from!
    for token in streaming_response.response_gen:
        # Filter out symbols that break formatting
        if token == "$":
            # Escape the $ sign, otherwise LaTeX formatting will be triggered!
            yield f"\\${token}"
        else:
            yield token


def save_graph_visualisation(_kgindex: KnowledgeGraphIndex) -> str:
    """Save the graph visualisation to a HTML file."""
    # Try to make a graph from the index at all depths
    graph = nx.Graph()
    already_added_nodes = set()
    nodes = list(_kgindex.index_struct.table.keys())[
        slice(st.session_state.settings_kgvis__max_nodes)
    ]

    for node in nodes:
        triplets = _kgindex.storage_context.graph_store.get(node)
        if node not in already_added_nodes:
            node_title = node.replace("_", " ").capitalize()
            graph.add_node(
                node, label=node_title, title=node_title, value=len(triplets) + 1
            )
            already_added_nodes.add(node)
        for triplet in triplets:
            # print(f"Edge: ({node}) - [{triplet[0]}] -> {triplet[1]}")
            if triplet[1] in nodes:
                edge_title = triplet[0].replace("_", " ").capitalize()
                graph.add_edge(node, triplet[1], title=edge_title)
    # Which layout is better?
    match st.session_state.settings_kgvis__layout:
        case "circular":
            positions = nx.circular_layout(graph, scale=len(nodes) * 24)
        case "planar":
            positions = nx.planar_layout(graph, scale=len(nodes) * 24)
        case "shell":
            positions = nx.shell_layout(graph, scale=len(nodes) * 24)
        case "spectral":
            positions = nx.spectral_layout(graph, scale=len(nodes) * 24)
        case "spring":
            positions = nx.spring_layout(graph, scale=len(nodes) * 24)
        case _:
            positions = nx.spring_layout(graph, scale=len(nodes) * 24)

    network = Network(
        notebook=False,
        height=f"{st.session_state.settings_kgvis__height}px",
        width="100%",
        # cdn_resources="in_line",
        neighborhood_highlight=True,
        directed=True,
    )
    network.toggle_physics(st.session_state.settings_kgvis__physics_enabled)
    network.toggle_hide_edges_on_drag(True)
    for node in graph.nodes:
        network.add_node(
            node,
            label=graph.nodes[node]["label"],
            title=graph.nodes[node]["title"],
            x=positions[node][0],
            y=positions[node][1],
            value=graph.nodes[node]["value"],
            mass=graph.nodes[node]["value"],
        )
    for edge in graph.edges:
        network.add_edge(edge[0], edge[1], title=graph.edges[edge]["title"])
    network.save_graph(st.session_state.settings_kgvis__filename)


@st.cache_resource(show_spinner=False)
def build_index(_documents, _clear_existing_kg=False):
    """Build the knowledge graph index from the documents. Method is cached in Streamlit."""
    if _clear_existing_kg:
        st.warning(":bathtub: Clearing existing knowledge graphs...")
        st.session_state.graph_store.query(
            """
            MATCH (n) DETACH DELETE n
            """
        )
    else:
        st.warning(
            ":bathtub: Keeping, perhaps overwriting, existing knowledge graphs..."
        )
    st.warning(":sheep: Parsing chunks from sources...")
    chunk_parser = SentenceSplitter.from_defaults(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True,
    )
    chunk_nodes = chunk_parser.get_nodes_from_documents(
        documents=_documents, show_progress=True
    )
    st.warning(f":ram: Added {len(chunk_nodes)} chunks to the document store.")
    st.warning(
        f":snail: Building index using {st.session_state.settings_llm__model} LLM. This WILL take time, unless it is read from the cache. Sit back and relax!"
    )
    kg_idx = KnowledgeGraphIndex(
        nodes=chunk_nodes,
        llm=st.session_state.client,
        embed_model=Settings.embed_model,
        storage_context=st.session_state.storage_context,
        max_triplets_per_chunk=st.session_state.settings_llm__index_max_triplets_per_chunk,
        include_embeddings=st.session_state.settings_llm__index_include_embeddings,
        show_progress=True,
    )
    kg_idx.storage_context.docstore.add_documents(chunk_nodes)
    return kg_idx


def initialise_chat_engine():
    """Initialise the chat engine with the index if it has been initialised."""
    post_processors = None
    if st.session_state.settings_llm__provider == const_llm_providers.cohere:
        post_processors = [
            CohereRerank(api_key=st.session_state.settings_llm__cohere_api_key)
        ]
    if "index" in st.session_state.keys() and st.session_state.index is not None:
        if "memory" in st.session_state.keys():
            st.session_state.memory.reset()
        return st.session_state.index.as_chat_engine(
            chat_mode=st.session_state.settings_llm__index_chat_mode,
            llm=st.session_state.client,
            verbose=True,
            memory=st.session_state.memory,
            system_prompt=st.session_state.settings_llm__llm_system_message,
            node_postprocessors=post_processors,
        )
    return None


def clear_chat_history():
    """Clear the chat history."""
    if "memory" in st.session_state.keys():
        st.session_state.memory.reset()
    st.session_state.messages.clear()


def export_chat_history_as_json():
    """Export the chat history to a file."""
    if len(st.session_state.messages) > 0:
        with container_data_source.expander(
            f"**Exported chat history of {len(st.session_state.messages)} messages**",
            expanded=True,
        ):
            with st.container(height=300, border=0):
                st.json(
                    json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
                )


# Main page content

col1_main, col2_main = st.columns(2, gap="large")

container_data_source = col1_main.container()

container_data_source.markdown("## Data source")

container_data_source.radio(
    "Source type",
    ["Wikipedia", "arXiv", "Pubmed", "Web page", "PDF"],
    captions=[
        "Wikipedia article",
        "arXiv query",
        "Pubmed query",
        "URL of a webpage",
        "URL of a PDF file",
    ],
    help="Select the type of source document to fetch. For arXiv and Pubmed, a search query may result in many relevant articles. Only the top 10 will be used as input sources.",
    key="ui__radio_source_type",
    horizontal=True,
    on_change=require_force_index_rebuild,
)


def format_wikipedia_languages(option):
    """Format the Wikipedia language options in a `{prefix}: {language name}` format."""
    return f"{option}: {wikipedia.languages()[option]}"


col1_wikipedia_prefix, col2_webpage_reader = container_data_source.columns(2)

col1_wikipedia_prefix.selectbox(
    "Wikipedia language prefix",
    # key="ui__select_webpage_reader",
    options=list(wikipedia.languages().keys()),
    format_func=format_wikipedia_languages,
    key="ui__selectbox_wikipedia_prefix",
    disabled=st.session_state.ui__radio_source_type != "Wikipedia",
    help="Select the Wikipedia language prefix (defaults to `en: English`). Check supported languages here: [List of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).",
)

col2_webpage_reader.selectbox(
    "Web page reader",
    key="ui__select_webpage_reader",
    options=["BeautifulSoup", "Trafilatura"],
    disabled=st.session_state.ui__radio_source_type != "Web page",
    help="Select the type of reader to use to extract text from the web page. _Only available for selection if you use `Web page` as the input source._ One reader maybe more efficient than another in the task of extracting text, depending on the source web page.",
)

container_data_source.text_input(
    "Source documents or existing index ID",
    key="ui__txtinput_document_source",
    help="""
    Input sources can be Wikipedia article reference, arXiv paper ID or query, PubMed query, a web page or a PDF. Additionally, you can provide an existing index ID to load an existing knowledge graph index from Redis.

    For **Wikipedia**, add the article reference below. For example, the reference for \'Artificial Intelligence\' is `Artificial_intelligence` (See [the article in English](https://en.wikipedia.org/wiki/Artificial_intelligence)).
    For **arXiv** articles, provide the paper ID or search query. For example, the paper ID for \'Quantum Locality in Game Strategy\' is `1609.06389` (See [the paper on arXiv](https://arxiv.org/abs/1609.06389)).
    For **Pubmed** articles, provide a search query. 
    For both arXiv and Pubmed, a search query may result in many relevant articles. Only the top 10 will be used as input sources.
    For a **web page**, provide its URL. Note that anything other than HTML for a web page _may not be supported_.
    For a **PDF**, provide its URL. The PDF will be downloaded and its text will be extracted.

    Note: 
    1. :red[To load an **existing knowledge graph index**, provide its index ID from Redis.]
    2. :red[Once an index is in memory after being built or loaded, the index ID will be shown here.]

    """,
    on_change=require_force_index_rebuild,
)


def _md5_hash(some_string):
    return hashlib.md5(some_string.encode("utf-8")).hexdigest()


def load_remote_pdf_data(pdf_url: str):
    """Download the PDF from the URL and load its data as text."""
    pdf_response = requests.get(pdf_url, timeout=30)
    pdf_filename = f"__tmp-{_md5_hash(pdf_url)}.pdf"
    with open(pdf_filename, "wb") as pdf:
        pdf.write(pdf_response.content)
        pdf.close()
    reader = PyMuPDFReader()
    pdf_docs = reader.load_data(file_path=pdf_filename)
    # Remove the temporary file
    os.remove(pdf_filename)
    return pdf_docs


def fetch_documents_build_index():
    """Fetch the source document(s) and build the index."""
    try:
        # Clear any existing knowledge graph visualisation
        with container_data_source.status(
            "Reading article and digesting the information...", expanded=True
        ):
            with st.container(height=300, border=0):
                start_time = time.time()
                if st.session_state.ui__radio_source_type == "Wikipedia":
                    st.toast(
                        f":notebook: Looking for the {st.session_state.ui__txtinput_document_source} on Wikipedia in {format_wikipedia_languages(st.session_state.ui__selectbox_wikipedia_prefix)}."
                    )
                    reader = WikipediaReader()
                    if (
                        st.session_state.ui__selectbox_wikipedia_prefix
                        != const_default_wikipedia_language_prefix
                    ):
                        # A non-English Wikipedia language prefix is selected
                        documents = reader.load_data(
                            pages=[st.session_state.ui__txtinput_document_source],
                            lang_prefix=st.session_state.ui__selectbox_wikipedia_prefix,
                            auto_suggest=False,
                        )
                    else:
                        documents = reader.load_data(
                            pages=[st.session_state.ui__txtinput_document_source],
                            auto_suggest=False,
                        )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} Wikipedia article entries. Excerpts: {documents[0].doc_id}\n\r > {documents[0].get_text()[:Settings.chunk_size]}..."
                    )
                elif st.session_state.ui__radio_source_type == "arXiv":
                    reader = ArxivReader()
                    documents = reader.load_data(
                        papers_dir="tmp-arxiv",
                        search_query=st.session_state.ui__txtinput_document_source,
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} entries from multiple arXiv papers."
                    )
                elif st.session_state.ui__radio_source_type == "Pubmed":
                    reader = PubmedReader()
                    documents = reader.load_data(
                        search_query=st.session_state.ui__txtinput_document_source
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} entries from multiple Pubmed articles."
                    )
                elif st.session_state.ui__radio_source_type == "Web page":
                    match st.session_state.ui__select_webpage_reader:
                        case "Trafilatura":
                            reader = TrafilaturaWebReader()
                        case "BeautifulSoup":
                            reader = BeautifulSoupWebReader()
                    documents = reader.load_data(
                        urls=[st.session_state.ui__txtinput_document_source]
                    )
                    st.warning(
                        f":newspaper: Fetched the web page {st.session_state.ui__txtinput_document_source} using the {st.session_state.ui__select_webpage_reader} reader. Excerpts: {documents[0].doc_id}\n\r > {documents[0].get_text()[:Settings.chunk_size]}..."
                    )
                elif st.session_state.ui__radio_source_type == "PDF":
                    documents = load_remote_pdf_data(
                        pdf_url=st.session_state.ui__txtinput_document_source
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} pages(s) from {st.session_state.ui__txtinput_document_source}. Excerpts from the first page {documents[0].doc_id}\n\r > {documents[0].get_text()[:Settings.chunk_size]}..."
                    )
                else:
                    st.error(
                        f":warning: {st.session_state.ui__radio_source_type} data source is not yet supported. Please select another source."
                    )
                    return
                st.session_state.index = build_index(documents)
                if st.session_state.index is not None:
                    st.session_state.ui__txtinput_document_source = (
                        st.session_state.index.index_id
                    )
                st.session_state.ui__chk_source_rebuild_index = False
                st.warning(":speech_balloon: Initialising chat engine...")
                st.session_state.chatbot = initialise_chat_engine()
                end_time = time.time()
                st.success(
                    f":white_check_mark: Phew, done in {round(end_time-start_time)} second(s)! Now, you can chat about this data source."
                )
                st.info(
                    f":pig_nose: Saved index ID :red[{st.session_state.index.index_id}] to storage. Make a note of this ID for future reference to reload the index."
                )
    except Exception as build_index_e:
        st.error(
            f":x: Error while building index from document(s). {type(build_index_e).__name__}: {build_index_e}"
        )


def load_existing_index():
    """Load the existing index from the storage."""
    try:
        # Clear any existing knowledge graph visualisation
        with container_data_source.status("Loading existing index...", expanded=True):
            with st.container(height=300, border=0):
                start_time = time.time()
                st.warning(":bee: Loading index from Redis storage.")
                st.session_state.index = load_index_from_storage(
                    storage_context=st.session_state.storage_context,
                    index_id=st.session_state.ui__txtinput_document_source,
                )
                st.session_state.ui__chk_source_rebuild_index = False
                st.warning(":speech_balloon: Initialising chat engine...")
                st.session_state.chatbot = initialise_chat_engine()
                end_time = time.time()
                st.success(
                    f":white_check_mark: Done in {round(end_time-start_time)} second(s)! Loaded index {st.session_state.index.index_id} starting with '{list(st.session_state.index.index_struct.to_dict().get('table').keys())[0].title()}'! Now, you can chat about this data source."
                )
    except Exception as load_index_e:
        st.error(
            f":x: Error while building index from document(s). {type(load_index_e).__name__}: {load_index_e}"
        )


def generate_knowledge_graph_visualisation():
    """Generate the knowledge graph visualisation."""
    try:
        with container_data_source.status(
            "Creating graph visualisation...", expanded=True
        ):
            with st.container(height=300, border=0):
                start_time = time.time()
                st.warning(
                    f":snail: Building graph visualisation from {len(list(st.session_state.index.index_struct.table.keys()))} nodes. This MAY take time, if the graph is large!"
                )
                save_graph_visualisation(st.session_state.index)
                end_time = time.time()
                st.success(
                    f":white_check_mark: Done in {round(end_time-start_time)} second(s)! The graph visualisation can be seen in the [Knowledge Graph Visualisation](Knowledge_Graph_Visualisation) page in the sidebar on the left."
                )
    except Exception as gen_kgvis_e:
        st.error(
            f":x: Error while showing knowledge graph. {type(gen_kgvis_e).__name__}: {gen_kgvis_e}"
        )


col1_source, col2_source, col3_source, col4_export, col5_delete = (
    container_data_source.columns([1, 2, 2, 2, 1])
)

col1_source.button(
    "**Build** index",
    on_click=fetch_documents_build_index,
    help="Fetch the source document(s) and build the knowledge graph index.",
)

container_data_source.checkbox(
    "Clear the cache and rebuild index",
    key="ui__chk_source_rebuild_index",
    help="Check this box to clear the cache and rebuild index when the source document(s) have changed or you have changed the LLM.",
)

col2_source.button(
    "**Load** existing index, if using storage",
    disabled=st.session_state.ui__txtinput_document_source == ""
    or st.session_state.settings_graphdb__disable
    or st.session_state.settings_redis__disable,
    on_click=load_existing_index,
    help=f"{':warning: Either the knowledge graph, the index or documents are stored in-memory with no persistence to disk. You cannot load existing indices. To be able to load indices, use persistable graph, index and document storage.' if st.session_state.settings_graphdb__disable or st.session_state.settings_redis__disable else 'Load the existing knowledge graph index from the storage.'}",
)

col3_source.button(
    "**Visualise** knowledge graph",
    disabled=("index" not in st.session_state.keys() or st.session_state.index is None),
    on_click=generate_knowledge_graph_visualisation,
    help="Visualise the knowledge graph about the current source.",
)

if len(st.session_state.messages) > 0:
    col4_export.button(
        "**Export** chat history as JSON", on_click=export_chat_history_as_json
    )
    col5_delete.button(
        "**Clear** chat",
        on_click=clear_chat_history,
    )


# Chatbot
# col1_heading, col2_export, col3_delete = col2_main.columns([3, 2, 1])
col2_main.markdown("## Chat with the data")

if "chatbot" in st.session_state.keys() and st.session_state.chatbot is not None:
    container_chat = col2_main.container(height=400, border=1)

    for message in st.session_state.messages:  # Display the prior chat messages
        with container_chat.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := col2_main.chat_input("Type your question or message here"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "timestamp": f"{datetime.datetime.now()}",
            }
        )
        with container_chat.chat_message("user"):
            st.markdown(prompt)
        if (
            len(st.session_state.messages) > 0
            and st.session_state.messages[-1]["role"] != "assistant"
        ):
            with container_chat.chat_message("assistant"):
                with st.spinner("Looking for responses..."):
                    response = st.write_stream(
                        stream_wrapper(st.session_state.chatbot.stream_chat(prompt))
                    )
            # Add response to message history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": f"{datetime.datetime.now()}",
                }
            )
else:
    col2_main.info(
        """:information_desk_person: Chatbot is not ready! Please build the index 
        from the source document(s) or reload an existing index to start chatting."""
    )


if st.session_state.langfuse_callback_handler is not None:
    st.session_state.langfuse_callback_handler.flush()
