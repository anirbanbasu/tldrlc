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

"""Solara global reactive state variables."""

import time
import solara
import os
from dotenv import load_dotenv
import hashlib

import logging

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.storage.chat_store import BaseChatStore
from llama_index.core.memory import ChatMemoryBuffer

from typing import Any, List
from typing_extensions import TypedDict

import utils.constants as constants

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama


from langfuse import Langfuse
from utils.callbacks import TLDRLCLangfuseCallbackHandler

logger = logging.getLogger(__name__)

logging.basicConfig(
    format=constants.LOG_FORMAT,
    level=int(
        os.getenv(constants.ENV_KEY_LOG_LEVEL, constants.DEFAULT_SETTING_LOG_LEVEL)
    ),
    encoding=constants.CHAR_ENCODING_UTF8,
)

status_message: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_colour: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_show: solara.Reactive[bool] = solara.reactive(False)


def show_status_message(message: str, colour: str = "info", timeout: int = 4):
    """Show a status message on the page."""
    status_message.value = message
    status_message_colour.value = colour
    status_message_show.value = True
    if timeout > 0:
        time.sleep(timeout)
        status_message_show.value = False


""" General settings """
global_settings_initialised: solara.Reactive[bool] = solara.reactive(False)

""" Language model settings """
global_settings__language_model_provider: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__cohere_api_key: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__cohere_model: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__openai_model: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__openai_api_key: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__ollama_url: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__ollama_model: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__llm_temperature: solara.Reactive[float] = solara.reactive(0.0)
global_settings__llm_chunk_size: solara.Reactive[int] = solara.reactive(0)
global_settings__llm_chunk_overlap: solara.Reactive[int] = solara.reactive(0)
global_settings__llm_request_timeout: solara.Reactive[int] = solara.reactive(0)
global_settings__llm_system_message: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

""" Index and chat settings """

global_settings__index_memory_token_limit: solara.Reactive[int] = solara.reactive(0)
global_settings__index_max_triplets_per_chunk: solara.Reactive[int] = solara.reactive(0)
global_settings__index_include_embeddings: solara.Reactive[bool] = solara.reactive(
    False
)
global_settings__index_chat_mode: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

""" Neo4j settings """
global_settings__neo4j_disable: solara.Reactive[bool] = solara.reactive(False)
global_settings__neo4j_url: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__neo4j_username: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__neo4j_password: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__neo4j_db_name: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

""" Redis settings """
global_settings__redis_disable: solara.Reactive[bool] = solara.reactive(False)
global_settings__redis_url: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__redis_namespace: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

""" Knowledge graph visualisation settings """
global_settings__kg_vis_height: solara.Reactive[int] = solara.reactive(0)
global_settings__kg_vis_max_nodes: solara.Reactive[int] = solara.reactive(0)
global_settings__kg_vis_max_depth: solara.Reactive[int] = solara.reactive(0)
global_settings__kg_vis_layout: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__kg_vis_physics_enabled: solara.Reactive[bool] = solara.reactive(False)

""" LlamaIndex Settings objects """
global_llamaindex_storage_context: solara.Reactive[StorageContext] = solara.reactive(
    None
)
global_llamaindex_chat_store: solara.Reactive[BaseChatStore] = solara.reactive(None)
global_llamaindex_chat_memory: solara.Reactive[ChatMemoryBuffer] = solara.reactive(None)

""" Chatbot objects """


class MessageDict(TypedDict):
    role: str
    content: str
    timestamp: str
    llm_provider: str = None
    llm_model_name: str = None


global_knowledge_graph_index: solara.Reactive[KnowledgeGraphIndex] = solara.reactive(
    None
)
global_chat_engine: solara.Reactive[BaseChatEngine] = solara.reactive(None)
global_chat_messages: solara.Reactive[List[MessageDict]] = solara.reactive([])


def md5_hash(some_string):
    return hashlib.md5(some_string.encode(constants.CHAR_ENCODING_UTF8)).hexdigest()


def setup_langfuse():
    """Setup (or, disable) Langfuse for performance evaluation."""
    use_langfuse = bool(
        os.getenv(
            constants.ENV_KEY_EVAL_USE_LANGFUSE,
            constants.DEFAULT_SETTING_EVAL_USE_LANGFUSE,
        ).lower()
        in ["true", "yes", "t", "y", "on"]
    )
    if use_langfuse:
        langfuse__secret_key = os.getenv(constants.ENV_KEY_LANGFUSE_SECRET_KEY)
        langfuse__public_key = os.getenv(constants.ENV_KEY_LANGFUSE_PUBLIC_KEY)
        langfuse__host = os.getenv(constants.ENV_KEY_LANGFUSE_HOST)
        try:
            langfuse = Langfuse(
                public_key=langfuse__public_key,
                secret_key=langfuse__secret_key,
                host=langfuse__host,
            )
            langfuse.auth_check()
            # Setup LangFuse tags for the handler, if necessary
            env__langfuse_trace_tags = os.getenv(
                constants.ENV_KEY_LANGFUSE_TRACE_TAGS, None
            )
            langfuse_trace_tags = None
            if env__langfuse_trace_tags is not None:
                langfuse_trace_tags = [
                    x.strip() for x in env__langfuse_trace_tags.split(",")
                ]
            # Setup the callback handler
            langfuse_callback_handler = TLDRLCLangfuseCallbackHandler(
                secret_key=langfuse__secret_key,
                public_key=langfuse__public_key,
                host=langfuse__host,
                tags=langfuse_trace_tags,
            )
            Settings.callback_manager = CallbackManager([langfuse_callback_handler])
            logger.warning(
                f"Using Langfuse at {langfuse__host} for performance evaluation."
            )
        except Exception as langfuse_e:
            logger.error(f"{langfuse_e} Langfuse setup failed. Disabling Langfuse.")
            Settings.callback_manager = None
    else:
        Settings.callback_manager = None
        logger.warning("Not using Langfuse for performance evaluation.")


def update_llm_settings(callback_data: Any = None):
    """Update the global LlamaIndex settings based on the user inputs."""
    match global_settings__language_model_provider.value:
        case constants.LLM_PROVIDER_COHERE:
            Settings.llm = Cohere(
                api_key=global_settings__cohere_api_key.value,
                model=global_settings__cohere_model.value,
                temperature=global_settings__llm_temperature.value,
                system_prompt=global_settings__llm_system_message.value,
            )
            Settings.embed_model = CohereEmbedding(
                cohere_api_key=global_settings__cohere_api_key.value,
                input_type="search_query",
            )
        case constants.LLM_PROVIDER_OPENAI:
            Settings.llm = OpenAI(
                model=global_settings__openai_model.value,
                temperature=global_settings__llm_temperature.value,
                system_prompt=global_settings__llm_system_message.value,
            )
            Settings.embed_model = OpenAIEmbedding()
        case constants.LLM_PROVIDER_OLLAMA:
            Settings.llm = Ollama(
                model=global_settings__ollama_model.value,
                request_timeout=global_settings__llm_request_timeout.value,
                base_url=global_settings__ollama_url.value,
                temperature=global_settings__llm_temperature.value,
                system_prompt=global_settings__llm_system_message.value,
            )
            Settings.embed_model = OllamaEmbedding(
                model_name=global_settings__ollama_model.value,
                base_url=global_settings__ollama_url.value,
            )
    Settings.chunk_size = global_settings__llm_chunk_size.value
    Settings.chunk_overlap = global_settings__llm_chunk_overlap.value


def update_chatbot_settings(callback_data: Any = None):
    if global_llamaindex_chat_store.value is None:
        global_llamaindex_chat_store.value = SimpleChatStore()

    if global_llamaindex_chat_memory.value is None:
        global_llamaindex_chat_memory.value = ChatMemoryBuffer.from_defaults(
            token_limit=global_settings__index_memory_token_limit.value,
            chat_store=global_llamaindex_chat_store.value,
        )
    else:
        global_llamaindex_chat_memory.value.token_limit = (
            global_settings__index_memory_token_limit.value
        )


def update_graph_storage_context(gs: Neo4jGraphStore = None):
    if not global_settings__neo4j_disable.value:
        if gs is None:
            gs = Neo4jGraphStore(
                username=global_settings__neo4j_username.value,
                password=global_settings__neo4j_password.value,
                url=global_settings__neo4j_url.value,
                database=global_settings__neo4j_db_name.value,
            )
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                graph_store=gs
            )
        else:
            global_llamaindex_storage_context.value.graph_store = gs
    else:
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                graph_store=SimpleGraphStore()
            )
        else:
            global_llamaindex_storage_context.value.graph_store = SimpleGraphStore()


def update_index_documents_storage_context():
    if not global_settings__redis_disable.value:
        kv_store = RedisKVStore(
            redis_uri=global_settings__redis_url.value,
        )
        document_store = RedisDocumentStore(
            redis_kvstore=kv_store,
            namespace=global_settings__redis_namespace.value,
        )
        index_store = RedisIndexStore(
            redis_kvstore=kv_store,
            namespace=global_settings__redis_namespace.value,
        )
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                docstore=document_store,
                index_store=index_store,
            )
        else:
            global_llamaindex_storage_context.value.docstore = document_store
            global_llamaindex_storage_context.value.index_store = index_store
    else:
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
            )
        else:
            global_llamaindex_storage_context.value.docstore = SimpleDocumentStore()
            global_llamaindex_storage_context.value.index_store = SimpleIndexStore()


def initialise_default_settings():
    """Load the global settings from the environment variables."""
    if not global_settings_initialised.value:
        """ Load the environment variables from the .env file, if present. """
        load_dotenv()

        """ Language model settings """
        global_settings__language_model_provider.value = os.getenv(
            constants.ENV_KEY_LLM_PROVIDER, constants.DEFAULT_SETTING_LLM_PROVIDER
        )

        global_settings__cohere_api_key.value = os.getenv(
            constants.ENV_KEY_COHERE_API_KEY, None
        )
        global_settings__cohere_model.value = os.getenv(
            constants.ENV_KEY_COHERE_MODEL, constants.DEFAULT_SETTING_COHERE_MODEL
        )
        global_settings__openai_model.value = os.getenv(
            constants.ENV_KEY_OPENAI_MODEL, constants.DEFAULT_SETTING_OPENAI_MODEL
        )
        global_settings__openai_api_key.value = os.getenv(
            constants.ENV_KEY_OPENAI_API_KEY, None
        )
        global_settings__ollama_url.value = os.getenv(
            constants.ENV_KEY_OLLAMA_URL, constants.DEFAULT_SETTING_OLLAMA_URL
        )
        global_settings__ollama_model.value = os.getenv(
            constants.ENV_KEY_OLLAMA_MODEL, constants.DEFAULT_SETTING_OLLAMA_MODEL
        )
        global_settings__llm_temperature.value = float(
            os.getenv(
                constants.ENV_KEY_LLM_TEMPERATURE,
                constants.DEFAULT_SETTING_LLM_TEMPERATURE,
            )
        )
        global_settings__llm_chunk_size.value = int(
            os.getenv(
                constants.ENV_KEY_LLM_CHUNK_SIZE,
                constants.DEFAULT_SETTING_LLM_CHUNK_SIZE,
            )
        )
        global_settings__llm_chunk_overlap.value = int(
            os.getenv(
                constants.ENV_KEY_LLM_CHUNK_OVERLAP,
                constants.DEFAULT_SETTING_LLM_CHUNK_OVERLAP,
            )
        )
        global_settings__llm_request_timeout.value = int(
            os.getenv(
                constants.ENV_KEY_LLM_REQUEST_TIMEOUT,
                constants.DEFAULT_SETTING_LLM_REQUEST_TIMEOUT,
            )
        )
        global_settings__llm_system_message.value = os.getenv(
            constants.ENV_KEY_LLM_SYSTEM_MESSAGE,
            constants.DEFAULT_SETTING_LLM_SYSTEM_MESSAGE,
        )

        """ Index and chat settings """

        global_settings__index_memory_token_limit.value = int(
            os.getenv(
                constants.ENV_KEY_INDEX_MEMORY_TOKEN_LIMIT,
                constants.DEFAULT_SETTING_INDEX_MEMORY_TOKEN_LIMIT,
            )
        )
        global_settings__index_max_triplets_per_chunk.value = int(
            os.getenv(
                constants.ENV_KEY_INDEX_MAX_TRIPLETS_PER_CHUNK,
                constants.DEFAULT_SETTING_INDEX_MAX_TRIPLETS_PER_CHUNK,
            )
        )
        global_settings__index_include_embeddings.value = bool(
            os.getenv(
                constants.ENV_KEY_INDEX_INCLUDE_EMBEDDINGS,
                constants.DEFAULT_SETTING_INDEX_INCLUDE_EMBEDDINGS,
            ).lower()
            in ["true", "yes", "t", "y", "on"]
        )
        global_settings__index_chat_mode.value = os.getenv(
            constants.ENV_KEY_INDEX_CHAT_MODE,
            constants.DEFAULT_SETTING_INDEX_CHAT_MODE,
        )

        """ Neo4j settings """
        global_settings__neo4j_disable.value = bool(
            os.getenv(
                constants.ENV_KEY_NEO4J_DISABLE, constants.DEFAULT_SETTING_NEO4J_DISABLE
            ).lower()
            in ["true", "yes", "t", "y", "on"]
        )
        global_settings__neo4j_url.value = os.getenv(
            constants.ENV_KEY_NEO4J_URL, constants.DEFAULT_SETTING_NEO4J_URL
        )
        global_settings__neo4j_username.value = os.getenv(
            constants.ENV_KEY_NEO4J_USERNAME, constants.DEFAULT_SETTING_NEO4J_USERNAME
        )
        global_settings__neo4j_password.value = os.getenv(
            constants.ENV_KEY_NEO4J_PASSWORD, None
        )
        global_settings__neo4j_db_name.value = os.getenv(
            constants.ENV_KEY_NEO4J_DB_NAME, constants.DEFAULT_SETTING_NEO4J_DB_NAME
        )

        """ Redis settings """
        global_settings__redis_disable.value = bool(
            os.getenv(
                constants.ENV_KEY_REDIS_DISABLE, constants.DEFAULT_SETTING_REDIS_DISABLE
            ).lower()
            in ["true", "yes", "t", "y", "on"]
        )
        global_settings__redis_url.value = os.getenv(
            constants.ENV_KEY_REDIS_URL, constants.DEFAULT_SETTING_REDIS_URL
        )
        global_settings__redis_namespace.value = os.getenv(
            constants.ENV_KEY_REDIS_NAMESPACE, constants.DEFAULT_SETTING_REDIS_NAMESPACE
        )

        """ Knowledge graph visualisation settings """
        global_settings__kg_vis_height.value = int(
            os.getenv(
                constants.ENV_KEY_KG_VIS_HEIGHT, constants.DEFAULT_SETTING_KG_VIS_HEIGHT
            )
        )
        global_settings__kg_vis_max_nodes.value = int(
            os.getenv(
                constants.ENV_KEY_KG_VIS_MAX_NODES,
                constants.DEFAULT_SETTING_KG_VIS_MAX_NODES,
            )
        )
        global_settings__kg_vis_max_depth.value = int(
            os.getenv(
                constants.ENV_KEY_KG_VIS_MAX_DEPTH,
                constants.DEFAULT_SETTING_KG_VIS_MAX_DEPTH,
            )
        )
        global_settings__kg_vis_layout.value = os.getenv(
            constants.ENV_KEY_KG_VIS_LAYOUT, constants.DEFAULT_SETTING_KG_VIS_LAYOUT
        )
        global_settings__kg_vis_physics_enabled.value = bool(
            os.getenv(
                constants.ENV_KEY_KG_VIS_PHYSICS_ENABLED,
                constants.DEFAULT_SETTING_KG_VIS_PHYSICS_ENABLED,
            ).lower()
            in ["true", "yes", "t", "y", "on"]
        )

        setup_langfuse()
        # Update all the settings and create objects to be used by other pages and components
        update_llm_settings()
        update_chatbot_settings()
        update_graph_storage_context()
        update_index_documents_storage_context()

        global_settings_initialised.value = True


def update_openai_apikey(callback_data: Any = None):
    """Update the OpenAI API key in the OS environment variable."""
    os.environ[constants.ENV_KEY_OPENAI_API_KEY] = global_settings__openai_api_key.value
    update_llm_settings()


def update_cohere_apikey(callback_data: Any = None):
    """Update the Cohere API key in the OS environment variable."""
    os.environ[constants.ENV_KEY_COHERE_API_KEY] = global_settings__cohere_api_key.value
    update_llm_settings()
