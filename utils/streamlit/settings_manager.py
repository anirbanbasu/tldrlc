# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=broad-exception-caught
# pylint: disable=duplicate-code

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

"""Module for managing settings of the application."""

import os
import types
from urllib.parse import urlparse
from dotenv import load_dotenv

import streamlit as st
from llama_index.core import Settings

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama

from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.callbacks import CallbackManager
from llama_index.core import StorageContext

from langfuse import Langfuse

# from langfuse.llama_index import LlamaIndexCallbackHandler
from utils.callbacks.ragas_langfuse import RagasLangfuseCallbackHandler


def initialise_constants():
    """Initialise the constants."""
    # Constants
    # Default Wikipedia language prefix
    st.session_state.const_default_wikipedia_language_prefix = "en"

    # Supported LLM providers
    st.session_state.const_llm_providers = types.SimpleNamespace()


# Session state metadata cannot be used from the other pages if they are connected to widgest.
# See: https://docs.streamlit.io/library/advanced-features/widget-behavior#save-widget-values-in-session-state-to-preserve-them-between-pages
def initialise_llm_settings():
    """Initialise the language model settings."""
    # LLM settings
    if "settings_llm__request_timeout" not in st.session_state:
        st.session_state.settings_llm__request_timeout = int(
            os.getenv("LLM_REQUEST_TIMEOUT", "120")
        )
    if "settings_llm__llm_temperature" not in st.session_state:
        st.session_state.settings_llm__llm_temperature = float(
            os.getenv("LLM_TEMPERATURE", "0.0")
        )
    if "settings_llm__llm_system_message" not in st.session_state:
        st.session_state.settings_llm__llm_system_message = os.getenv(
            "LLM_SYSTEM_MESSAGE",
            "You are an intelligent assistant. You respond to questions about articles from various sources such as Wikipedia, arXiv, Pubmed and so on. You generate your answers based ONLY on information in those articles that are provided to you as context information. If you are unsure about an answer or if the user query cannot be answered using information in the context then say that you do not know. If the user ask you to stop, immediately stop answering.",
        )
    if "settings_llm__llm_chunk_size" not in st.session_state:
        st.session_state.settings_llm__llm_chunk_size = int(
            os.getenv("LLM_CHUNK_SIZE", "1024")
        )
    if "settings_llm__llm_chunk_overlap" not in st.session_state:
        st.session_state.settings_llm__llm_chunk_overlap = int(
            os.getenv("LLM_CHUNK_OVERLAP", "64")
        )
    if "settings_llm__index_memory_token_limit" not in st.session_state:
        st.session_state.settings_llm__index_memory_token_limit = int(
            os.getenv("INDEX_MEMORY_TOKEN_LIMIT", "4096")
        )
    if "settings_llm__index_max_triplets_per_chunk" not in st.session_state:
        st.session_state.settings_llm__index_max_triplets_per_chunk = int(
            os.getenv("INDEX_MAX_TRIPLETS_PER_CHUNK", "8")
        )
    if "settings_llm__index_include_embeddings" not in st.session_state:
        st.session_state.settings_llm__index_include_embeddings = bool(
            os.getenv("INDEX_INCLUDE_EMBEDDINGS", "True").lower()
            in ["true", "yes", "t", "y", "on"]
        )
    if "settings_llm__index_chat_mode" not in st.session_state:
        st.session_state.settings_llm__index_chat_mode = os.getenv(
            "INDEX_CHAT_MODE", "context"
        )


def initialise_shadow_llm_settings():
    """Initialise the shadow session state variables for language model settings."""
    if "settings_llm__provider" in st.session_state:
        st.session_state.shadow__settings_llm__provider = (
            st.session_state.settings_llm__provider
        )
    if "settings_llm__request_timeout" in st.session_state:
        st.session_state.shadow__settings_llm__request_timeout = (
            st.session_state.settings_llm__request_timeout
        )
    if "settings_llm__llm_temperature" in st.session_state:
        st.session_state.shadow__settings_llm__llm_temperature = (
            st.session_state.settings_llm__llm_temperature
        )
    if "settings_llm__llm_system_message" in st.session_state:
        st.session_state.shadow__settings_llm__llm_system_message = (
            st.session_state.settings_llm__llm_system_message
        )
    if "settings_llm__llm_chunk_size" in st.session_state:
        st.session_state.shadow__settings_llm__llm_chunk_size = (
            st.session_state.settings_llm__llm_chunk_size
        )
    if "settings_llm__llm_chunk_overlap" in st.session_state:
        st.session_state.shadow__settings_llm__llm_chunk_overlap = (
            st.session_state.settings_llm__llm_chunk_overlap
        )
    if "settings_llm__index_memory_token_limit" in st.session_state:
        st.session_state.shadow__settings_llm__index_memory_token_limit = (
            st.session_state.settings_llm__index_memory_token_limit
        )
    if "settings_llm__index_max_triplets_per_chunk" in st.session_state:
        st.session_state.shadow__settings_llm__index_max_triplets_per_chunk = (
            st.session_state.settings_llm__index_max_triplets_per_chunk
        )
    if "settings_llm__index_include_embeddings" in st.session_state:
        st.session_state.shadow__settings_llm__index_include_embeddings = (
            st.session_state.settings_llm__index_include_embeddings
        )
    if "settings_llm__index_chat_mode" in st.session_state:
        st.session_state.shadow__settings_llm__index_chat_mode = (
            st.session_state.settings_llm__index_chat_mode
        )


def initialise_graphdb_settings():
    """Initialise the graph database settings."""
    # Graph database settings
    if "settings_graphdb__disable" not in st.session_state:
        st.session_state.settings_graphdb__disable = bool(
            os.getenv("NEO4J_DISABLE", "False").lower()
            in ["true", "yes", "t", "y", "on"]
        )
    if "settings_graphdb__url" not in st.session_state:
        st.session_state.settings_graphdb__url = os.getenv(
            "NEO4J_URL", "bolt://localhost:7687"
        )
    if "settings_graphdb__username" not in st.session_state:
        st.session_state.settings_graphdb__username = os.getenv(
            "NEO4J_USERNAME", "neo4j"
        )
    if "settings_graphdb__password" not in st.session_state:
        st.session_state.settings_graphdb__password = os.getenv("NEO4J_PASSWORD")
    if "settings_graphdb__dbname" not in st.session_state:
        st.session_state.settings_graphdb__dbname = os.getenv("NEO4J_DB_NAME", "neo4j")


def initialise_shadow_graphdb_settings():
    """Initialise the shadow session state variables for graph database settings."""
    if "settings_graphdb__disable" in st.session_state:
        st.session_state.shadow__settings_graphdb__disable = (
            st.session_state.settings_graphdb__disable
        )
    if "settings_graphdb__url" in st.session_state:
        st.session_state.shadow__settings_graphdb__url = (
            st.session_state.settings_graphdb__url
        )
    if "settings_graphdb__username" in st.session_state:
        st.session_state.shadow__settings_graphdb__username = (
            st.session_state.settings_graphdb__username
        )
    if "settings_graphdb__password" in st.session_state:
        st.session_state.shadow__settings_graphdb__password = (
            st.session_state.settings_graphdb__password
        )
    if "settings_graphdb__dbname" in st.session_state:
        st.session_state.shadow__settings_graphdb__dbname = (
            st.session_state.settings_graphdb__dbname
        )


def initialise_document_index_store_settings():
    """Initialise the document and index store settings."""
    # Redis settings
    if "settings_redis__disable" not in st.session_state:
        st.session_state.settings_redis__disable = bool(
            os.getenv("REDIS_DISABLE", "False").lower()
            in ["true", "yes", "t", "y", "on"]
        )
    if "settings_redis__url" not in st.session_state:
        st.session_state.settings_redis__url = os.getenv(
            "REDIS_URL", "redis://localhost:6379"
        )
    if "settings_redis__namespace" not in st.session_state:
        st.session_state.settings_redis__namespace = os.getenv(
            "REDIS_NAMESPACE", "tldrlc"
        )


def initialise_shadow_document_index_store_settings():
    """Initialise the shadow session state variables for document and index store settings."""
    if "settings_redis__disable" in st.session_state:
        st.session_state.shadow__settings_redis__disable = (
            st.session_state.settings_redis__disable
        )
    if "settings_redis__url" in st.session_state:
        st.session_state.shadow__settings_redis__url = (
            st.session_state.settings_redis__url
        )
    if "settings_redis__namespace" in st.session_state:
        st.session_state.shadow__settings_redis__namespace = (
            st.session_state.settings_redis__namespace
        )


def initialise_graph_visualisation_settings():
    """Initialise the graph visualisation settings."""
    # Knowledge graph visualisation settings
    if "settings_kgvis__height" not in st.session_state:
        st.session_state.settings_kgvis__height = int(os.getenv("KG_VIS_HEIGHT", "800"))
    if "settings_kgvis__max_nodes" not in st.session_state:
        st.session_state.settings_kgvis__max_nodes = int(
            os.getenv("KG_VIS_MAX_NODES", "100")
        )
    if "settings_kgvis__max_depth" not in st.session_state:
        st.session_state.settings_kgvis__max_depth = int(
            os.getenv("KG_VIS_MAX_DEPTH", "3")
        )
    if "settings_kgvis__layout" not in st.session_state:
        st.session_state.settings_kgvis__layout = os.getenv("KG_VIS_LAYOUT", "spring")
    if "settings_kgvis__physics_enabled" not in st.session_state:
        st.session_state.settings_kgvis__physics_enabled = bool(
            os.getenv("KG_VIS_PHYSICS_ENABLED", "True").lower()
            in ["true", "yes", "t", "y", "on"]
        )


def initialise_shadow_graph_visualisation_settings():
    """Initialise the shadow session state variables for graph visualisation settings."""
    if "settings_kgvis__height" in st.session_state:
        st.session_state.shadow__settings_kgvis__height = (
            st.session_state.settings_kgvis__height
        )
    if "settings_kgvis__filename" in st.session_state:
        st.session_state.shadow__settings_kgvis__filename = (
            st.session_state.settings_kgvis__filename
        )
    if "settings_kgvis__max_nodes" in st.session_state:
        st.session_state.shadow__settings_kgvis__max_nodes = (
            st.session_state.settings_kgvis__max_nodes
        )
    if "settings_kgvis__max_depth" in st.session_state:
        st.session_state.shadow__settings_kgvis__max_depth = (
            st.session_state.settings_kgvis__max_depth
        )
    if "settings_kgvis__layout" in st.session_state:
        st.session_state.shadow__settings_kgvis__layout = (
            st.session_state.settings_kgvis__layout
        )
    if "settings_kgvis__physics_enabled" in st.session_state:
        st.session_state.shadow__settings_kgvis__physics_enabled = (
            st.session_state.settings_kgvis__physics_enabled
        )


def initialise_shadow_widget_session_keys():
    """Initialise the shadow session state keys for the widgets."""
    if "ui__chk_source_rebuild_index" in st.session_state:
        st.session_state.shadow__ui__chk_source_rebuild_index = (
            st.session_state.ui__chk_source_rebuild_index
        )
    if "ui__txtinput_document_source" in st.session_state:
        st.session_state.shadow__ui__txtinput_document_source = (
            st.session_state.ui__txtinput_document_source
        )
    if "ui__radio_source_type" in st.session_state:
        st.session_state.shadow__ui__radio_source_type = (
            st.session_state.ui__radio_source_type
        )
    if "ui__selectbox_wikipedia_prefix" in st.session_state:
        st.session_state.shadow__ui__selectbox_wikipedia_prefix = (
            st.session_state.ui__selectbox_wikipedia_prefix
        )
    if "ui__select_webpage_reader" in st.session_state:
        st.session_state.shadow__ui__select_webpage_reader = (
            st.session_state.ui__select_webpage_reader
        )


def initialise_widget_session_keys():
    """Initialise the session state keys for the widgets."""
    # User interface backed-elements
    if "ui__chk_source_rebuild_index" not in st.session_state:
        st.session_state.ui__chk_source_rebuild_index = False

    if "ui__txtinput_document_source" not in st.session_state:
        if "index" not in st.session_state:
            st.session_state.ui__txtinput_document_source = os.getenv(
                "DEFAULT_SOURCE", None
            )
        else:
            st.session_state.ui__txtinput_document_source = (
                st.session_state.index.index_id
            )

    if "ui__radio_source_type" not in st.session_state:
        st.session_state.ui__radio_source_type = "Wikipedia"

    if "ui__selectbox_wikipedia_prefix" not in st.session_state:
        st.session_state.ui__selectbox_wikipedia_prefix = (
            st.session_state.const_default_wikipedia_language_prefix
        )

    if "ui__select_webpage_reader" not in st.session_state:
        st.session_state.ui__select_webpage_reader = "BeautifulSoup"


def initialise_settings():
    """Initialise the session state and other variables."""
    st.session_state.const_llm_providers.list = ["Cohere", "Ollama", "Open AI"]
    st.session_state.const_llm_providers.cohere = (
        st.session_state.const_llm_providers.list[0]
    )
    st.session_state.const_llm_providers.ollama = (
        st.session_state.const_llm_providers.list[1]
    )
    st.session_state.const_llm_providers.openai = (
        st.session_state.const_llm_providers.list[2]
    )

    # Mechanism to ensure that certain things run only once
    if "first_run" not in st.session_state:
        st.session_state.first_run = True

    # Setup performance evaluation using Langfuse
    if "use_langfuse" not in st.session_state:
        st.session_state.use_langfuse = bool(
            os.getenv("EVAL_USE_LANGFUSE", "True").lower()
            in ["true", "yes", "t", "y", "on"]
        )

        # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    initialise_llm_provider()
    initialise_llm_settings()
    initialise_graphdb_settings()
    initialise_document_index_store_settings()
    initialise_graph_visualisation_settings()
    initialise_widget_session_keys()


def initialise_shadow_settings():
    """Initialise the shadow session state variables."""
    initialise_shadow_llm_provider()
    initialise_shadow_llm_settings()
    initialise_shadow_graphdb_settings()
    initialise_shadow_document_index_store_settings()
    initialise_shadow_graph_visualisation_settings()
    initialise_shadow_widget_session_keys()


def setup_langfuse():
    """Setup (or, disable) Langfuse for performance evaluation."""
    if st.session_state.use_langfuse:
        langfuse__secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse__public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse__host = os.getenv("LANGFUSE_HOST")
        try:
            st.session_state.langfuse = Langfuse(
                public_key=langfuse__public_key,
                secret_key=langfuse__secret_key,
                host=langfuse__host,
            )
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
            st.error(f":x: {langfuse_e} Langfuse setup failed. Disabling Langfuse.")
            st.session_state.use_langfuse = False
            st.session_state.langfuse_callback_handler = None
    else:
        st.session_state.langfuse_callback_handler = None
        st.toast(":warning: Not using Langfuse for performance evaluation.")


def require_force_index_rebuild():
    """Force the index to be rebuilt when the source document(s) have changed."""
    st.session_state.shadow__ui__chk_source_rebuild_index = True
    st.cache_resource.clear()
    copy_shadow_ui_widget_session_keys()


def copy_shadow_ui_widget_session_keys():
    """Copy the shadow session state keys for the widgets."""
    st.session_state.ui__chk_source_rebuild_index = (
        st.session_state.shadow__ui__chk_source_rebuild_index
    )
    st.session_state.ui__txtinput_document_source = (
        st.session_state.shadow__ui__txtinput_document_source
    )
    st.session_state.ui__radio_source_type = (
        st.session_state.shadow__ui__radio_source_type
    )
    st.session_state.ui__selectbox_wikipedia_prefix = (
        st.session_state.shadow__ui__selectbox_wikipedia_prefix
    )
    st.session_state.ui__select_webpage_reader = (
        st.session_state.shadow__ui__select_webpage_reader
    )


def copy_shadow_openai_apikey():
    """Copy the shadow session state variables for OpenAI API key."""
    st.session_state.settings_llm__openai_api_key = (
        st.session_state.shadow__settings_llm__openai_api_key
    )


def update_openai_apikey():
    """Update the OpenAI API key in the OS environment variable."""
    copy_shadow_openai_apikey()
    os.environ["OPENAI_API_KEY"] = st.session_state.settings_llm__openai_api_key
    update_llm_settings()


def copy_shadow_cohere_apikey():
    """Copy the shadow session state variables for Cohere API key."""
    st.session_state.settings_llm__cohere_api_key = (
        st.session_state.shadow__settings_llm__cohere_api_key
    )


def update_cohere_apikey():
    """Update the Cohere API key in the OS environment variable."""
    copy_shadow_cohere_apikey()
    os.environ["COHERE_API_KEY"] = st.session_state.settings_llm__cohere_api_key
    update_llm_settings()


def copy_shadow_llm_provider_settings():
    """Copy the shadow session state variables for language model provider."""
    st.session_state.settings_llm__provider = (
        st.session_state.shadow__settings_llm__provider
    )
    st.session_state.settings_llm__model = st.session_state.shadow__settings_llm__model
    match st.session_state.shadow__settings_llm__provider:
        case st.session_state.const_llm_providers.cohere:
            copy_shadow_cohere_apikey()
            st.session_state.settings_llm__model_cohere = (
                st.session_state.settings_llm__model
            )
        case st.session_state.const_llm_providers.openai:
            copy_shadow_openai_apikey()
            st.session_state.settings_llm__model_openai = (
                st.session_state.settings_llm__model
            )
        case st.session_state.const_llm_providers.ollama:
            st.session_state.settings_llm__ollama_url = (
                st.session_state.shadow__settings_llm__ollama_url
            )
            st.session_state.settings_llm__model_ollama = (
                st.session_state.settings_llm__model
            )


def copy_shadow_llm_settings():
    """Copy the shadow session state variables for language model settings."""
    st.session_state.settings_llm__request_timeout = (
        st.session_state.shadow__settings_llm__request_timeout
    )
    st.session_state.settings_llm__llm_temperature = (
        st.session_state.shadow__settings_llm__llm_temperature
    )
    st.session_state.settings_llm__llm_system_message = (
        st.session_state.shadow__settings_llm__llm_system_message
    )
    st.session_state.settings_llm__llm_chunk_size = (
        st.session_state.shadow__settings_llm__llm_chunk_size
    )
    st.session_state.settings_llm__llm_chunk_overlap = (
        st.session_state.shadow__settings_llm__llm_chunk_overlap
    )
    st.session_state.settings_llm__index_memory_token_limit = (
        st.session_state.shadow__settings_llm__index_memory_token_limit
    )
    st.session_state.settings_llm__index_max_triplets_per_chunk = (
        st.session_state.shadow__settings_llm__index_max_triplets_per_chunk
    )
    st.session_state.settings_llm__index_include_embeddings = (
        st.session_state.shadow__settings_llm__index_include_embeddings
    )
    st.session_state.settings_llm__index_chat_mode = (
        st.session_state.shadow__settings_llm__index_chat_mode
    )


def switch_llm_provider():
    """Switch the language model provider."""
    st.session_state.settings_llm__provider = (
        st.session_state.shadow__settings_llm__provider
    )
    st.session_state.llm_provider_switched = True


def update_llm_settings():
    """Update the language model settings."""
    copy_shadow_llm_provider_settings()
    copy_shadow_llm_settings()
    try:
        match st.session_state.settings_llm__provider:
            case st.session_state.const_llm_providers.cohere:
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
            case st.session_state.const_llm_providers.openai:
                st.session_state.client = OpenAI(
                    model=st.session_state.settings_llm__model,
                    temperature=st.session_state.settings_llm__llm_temperature,
                )
                st.session_state.embed_model = OpenAIEmbedding()
            case st.session_state.const_llm_providers.ollama:
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
            f":white_check_mark: Using {st.session_state.settings_llm__provider} {st.session_state.settings_llm__model} language model. {'Cohere Reranker enabled.' if st.session_state.settings_llm__provider == st.session_state.const_llm_providers.cohere else ''}"
        )
        st.session_state.llm_provider_switched = False
    except Exception as llm_e:
        st.error(f":x: {type(llm_e).__name__}: {llm_e}")


def copy_shadow_graphdb_settings():
    """Copy the shadow session state variables for graph database settings."""
    st.session_state.settings_graphdb__disable = (
        st.session_state.shadow__settings_graphdb__disable
    )
    st.session_state.settings_graphdb__url = (
        st.session_state.shadow__settings_graphdb__url
    )
    st.session_state.settings_graphdb__username = (
        st.session_state.shadow__settings_graphdb__username
    )
    st.session_state.settings_graphdb__password = (
        st.session_state.shadow__settings_graphdb__password
    )


def update_graphdb_settings():
    """Update the graph database settings."""
    copy_shadow_graphdb_settings()
    if not st.session_state.settings_graphdb__disable:
        try:
            st.session_state.graph_store = Neo4jGraphStore(
                username=st.session_state.settings_graphdb__username,
                password=st.session_state.settings_graphdb__password,
                url=st.session_state.settings_graphdb__url,
                database=st.session_state.settings_graphdb__dbname,
            )
            if "storage_context" not in st.session_state:
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
        if "storage_context" not in st.session_state:
            st.session_state.storage_context = StorageContext.from_defaults(
                graph_store=SimpleGraphStore()
            )
        else:
            st.session_state.storage_context.graph_store = SimpleGraphStore()
        st.toast(
            ":warning: Neo4j graph database is disabled. Reverting to in-memory storage."
        )


def copy_shadow_document_index_store_settings():
    """Copy the shadow session state variables for document and index store settings."""
    st.session_state.settings_redis__disable = (
        st.session_state.shadow__settings_redis__disable
    )
    st.session_state.settings_redis__url = st.session_state.shadow__settings_redis__url
    st.session_state.settings_redis__namespace = (
        st.session_state.shadow__settings_redis__namespace
    )


def update_document_index_store_settings():
    """Update the document and index store settings."""
    copy_shadow_document_index_store_settings()
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
            if "storage_context" not in st.session_state:
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
        if "storage_context" not in st.session_state:
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


def copy_shadow_graph_visualisation_settings():
    """Copy the shadow session state variables for graph visualisation settings."""
    st.session_state.settings_kgvis__height = (
        st.session_state.shadow__settings_kgvis__height
    )
    st.session_state.settings_kgvis__max_nodes = (
        st.session_state.shadow__settings_kgvis__max_nodes
    )
    st.session_state.settings_kgvis__max_depth = (
        st.session_state.shadow__settings_kgvis__max_depth
    )
    st.session_state.settings_kgvis__layout = (
        st.session_state.shadow__settings_kgvis__layout
    )
    st.session_state.settings_kgvis__physics_enabled = (
        st.session_state.shadow__settings_kgvis__physics_enabled
    )


def update_graph_visualisation_settings():
    """Update the graph visualisation settings."""
    copy_shadow_graph_visualisation_settings()
    st.session_state.settings_saved_kgvis__height = (
        st.session_state.settings_kgvis__height
    )


def initialise_shadow_llm_provider():
    """Initialise the shadow session state variables for language model provider."""
    if "settings_llm__provider" in st.session_state:
        st.session_state.shadow__settings_llm__provider = (
            st.session_state.settings_llm__provider
        )
    if "settings_llm__cohere_api_key" in st.session_state:
        st.session_state.shadow__settings_llm__cohere_api_key = (
            st.session_state.settings_llm__cohere_api_key
        )
    if "settings_llm__openai_api_key" in st.session_state:
        st.session_state.shadow__settings_llm__openai_api_key = (
            st.session_state.settings_llm__openai_api_key
        )
    if "settings_llm__ollama_url" in st.session_state:
        st.session_state.shadow__settings_llm__ollama_url = (
            st.session_state.settings_llm__ollama_url
        )
    match st.session_state.shadow__settings_llm__provider:
        case st.session_state.const_llm_providers.cohere:
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_cohere
            )
        case st.session_state.const_llm_providers.openai:
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_openai
            )
        case st.session_state.const_llm_providers.ollama:
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_ollama
            )


def initialise_llm_provider():
    """Initialise the language model provider."""
    if "settings_llm__provider" not in st.session_state:
        st.session_state.settings_llm__provider = os.getenv("LLM_PROVIDER", "Ollama")
    if "settings_llm__cohere_api_key" not in st.session_state:
        st.session_state.settings_llm__cohere_api_key = os.getenv("COHERE_API_KEY")
    if "settings_llm__openai_api_key" not in st.session_state:
        st.session_state.settings_llm__openai_api_key = os.getenv("OPENAI_API_KEY")
    if "settings_llm__ollama_url" not in st.session_state:
        st.session_state.settings_llm__ollama_url = os.getenv(
            "OLLAMA_URL", "http://localhost:11434"
        )
    st.session_state.settings_llm__model_cohere = os.getenv("COHERE_MODEL", "command")
    st.session_state.settings_llm__model_openai = os.getenv(
        "OPENAI_MODEL", "gpt-3.5-turbo"
    )
    st.session_state.settings_llm__model_ollama = os.getenv("OLLAMA_MODEL", "mistral")


def load_settings():
    """Load the settings for the first time."""
    # Load environment variables from .env file, if present
    load_dotenv()
    initialise_constants()
    initialise_settings()
    initialise_shadow_settings()

    setup_langfuse()

    update_llm_settings()
    update_graphdb_settings()
    update_document_index_store_settings()
    update_graph_visualisation_settings()

    if st.session_state.first_run:
        st.session_state.first_run = False
