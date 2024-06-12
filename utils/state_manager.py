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
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from llama_index.core.storage.chat_store import BaseChatStore
from llama_index.core.memory import ChatMemoryBuffer

from typing import Any, List
import solara.lab
from typing_extensions import TypedDict

import utils.constants as constants

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore

# See: https://github.com/run-llama/llama_index/issues/10731#issuecomment-1946450169
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.llamafile import LlamafileEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.cohere import Cohere
from llama_index.llms.llamafile import Llamafile
from llama_index.llms.ollama import Ollama


from langfuse import Langfuse
from langfuse.decorators import langfuse_context
from utils.callbacks import TLDRLCLangfuseCallbackHandler

logger = logging.getLogger(__name__)

logging.basicConfig(
    format=constants.LOG_FORMAT,
    level=int(
        os.getenv(constants.ENV_KEY_LOG_LEVEL, constants.DEFAULT_SETTING_LOG_LEVEL)
    ),
    encoding=constants.CHAR_ENCODING_UTF8,
)

show_eu_ai_act_notice: solara.Reactive[bool] = solara.Reactive(True)

status_message: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_colour: solara.Reactive[str] = solara.reactive(constants.EMPTY_STRING)
status_message_show: solara.Reactive[bool] = solara.reactive(False)


def show_status_message(message: str, colour: str = "info", timeout: int = 4):
    """
    Update the reactive variables to be able to display a status message on a page. The
    message can be displayed in the form of a toast or banner.
    """
    status_message.value = message
    status_message_colour.value = colour
    status_message_show.value = True
    if timeout > 0:
        time.sleep(timeout)
        status_message_show.value = False


""" General settings """
global_settings_initialised: solara.Reactive[bool] = solara.reactive(False)
global_settings_langfuse_enabled: solara.Reactive[bool] = solara.reactive(False)
global_settings_langfuse_tags: solara.Reactive[List[str]] = solara.reactive(None)
global_client_langfuse_lowlevel: solara.Reactive[Langfuse] = solara.reactive(None)

""" Document ingestion pipeline cache """
global_cache__ingestion: solara.Reactive[RedisCache] = solara.reactive(None)

""" Language model settings """
global_settings__language_model_provider: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__llm_provider_notice: solara.Reactive[str] = solara.reactive(
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
global_settings__llamafile_url: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__ollama_url: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__ollama_model: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__embedding_model: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)
global_settings__llm_temperature: solara.Reactive[float] = solara.reactive(0.0)
global_settings__data_ingestion_chunk_size: solara.Reactive[int] = solara.reactive(0)
global_settings__data_ingestion_chunk_overlap: solara.Reactive[int] = solara.reactive(0)
global_settings__llm_request_timeout: solara.Reactive[int] = solara.reactive(0)
global_settings__llm_system_message: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)

""" Data ingestion settings """
global_settings__di_enable_title_extractor: solara.Reactive[bool] = solara.reactive(
    False
)
global_settings__di_enable_title_extractor_nodes: solara.Reactive[int] = (
    solara.reactive(5)
)
global_settings__di_enable_keyword_extractor: solara.Reactive[bool] = solara.reactive(
    False
)
global_settings__di_enable_keyword_extractor_keywords: solara.Reactive[int] = (
    solara.reactive(10)
)
global_settings__di_enable_qa_extractor: solara.Reactive[bool] = solara.reactive(False)
global_settings__di_enable_qa_extractor_questions: solara.Reactive[int] = (
    solara.reactive(3)
)
global_settings__di_enable_summary_extractor: solara.Reactive[bool] = solara.reactive(
    False
)
global_settings__di_enable_summary_extractor_summaries: solara.Reactive[List[str]] = (
    solara.reactive(["self", "prev"])
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

""" LlamaIndex Settings objects """
global_llamaindex_storage_context: solara.Reactive[StorageContext] = solara.reactive(
    None
)
global_llamaindex_chat_store: solara.Reactive[BaseChatStore] = solara.reactive(None)
global_llamaindex_chat_memory: solara.Reactive[ChatMemoryBuffer] = solara.reactive(None)


""" Chatbot objects """


class MessageDict(TypedDict):
    """A dictionary representing a chat message that is displayed to the user."""

    role: str
    content: str
    timestamp: str
    llm_provider: str = None
    llm_model_name: str = None


global_knowledge_graph_index: solara.Reactive[PropertyGraphIndex] = solara.reactive(
    None
)
global_semantic_search_index: solara.Reactive[VectorStoreIndex] = solara.reactive(None)
global_chat_engine: solara.Reactive[BaseChatEngine] = solara.reactive(None)
global_chat_messages: solara.Reactive[List[MessageDict]] = solara.reactive([])


def set_global_setting(
    setting: solara.Reactive,
    env_key: str,
    default_value: str = None,
    type_cast=str,
    convert_to_list=False,
    list_split_char=constants.SPACE_STRING,
):
    """
    Sets a global setting's value to the corresponding environment variable or a default value.

    Args:
        setting (solara.Reactive variable): The global setting to set.
        env_key (str): The key of the environment variable.
        default_value (str): The default value to use if the environment variable is not set. Defaults to None.
        type_cast (type): The type to cast the environment variable value to. Defaults to str.
        convert_to_list (bool): Whether to convert the cast value to a list. Defaults to False.
        list_split_char (str): The character to split the value into a list. Defaults to " ".
    """

    parsed_value = None
    if type_cast == bool:
        parsed_value = os.getenv(env_key, default_value).lower() in [
            "true",
            "yes",
            "t",
            "y",
            "on",
        ]
    else:
        parsed_value = os.getenv(env_key, default_value)

    setting.value = (
        type_cast(parsed_value)
        if not convert_to_list
        else [type_cast(v) for v in parsed_value.split(list_split_char)]
    )


def md5_hash(some_string):
    return hashlib.md5(some_string.encode(constants.CHAR_ENCODING_UTF8)).hexdigest()


def setup_langfuse():
    """Setup (or, disable) Langfuse for performance evaluation. Configure both decorator level and low-level API."""
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
            global_client_langfuse_lowlevel.value = Langfuse(
                public_key=langfuse__public_key,
                secret_key=langfuse__secret_key,
                host=langfuse__host,
            )
            global_client_langfuse_lowlevel.value.auth_check()
            # Setup LangFuse tags for the handler, if necessary
            env__langfuse_trace_tags = os.getenv(
                constants.ENV_KEY_LANGFUSE_TRACE_TAGS, None
            )
            if env__langfuse_trace_tags is not None:
                global_settings_langfuse_tags.value = [
                    x.strip() for x in env__langfuse_trace_tags.split(",")
                ]
            # Setup the callback handler
            langfuse_callback_handler = TLDRLCLangfuseCallbackHandler(
                secret_key=langfuse__secret_key,
                public_key=langfuse__public_key,
                host=langfuse__host,
                tags=global_settings_langfuse_tags.value,
            )
            Settings.callback_manager = CallbackManager([langfuse_callback_handler])
            global_settings_langfuse_enabled.value = True
            langfuse_context.configure(
                public_key=langfuse__public_key,
                secret_key=langfuse__secret_key,
                host=langfuse__host,
                enabled=True,
            )
            logger.warning(
                f"Using Langfuse at {langfuse__host} for performance evaluation."
            )
        except Exception as langfuse_e:
            Settings.callback_manager = None
            langfuse_context.configure(
                enabled=False,
            )
            global_settings_langfuse_enabled.value = False
            global_client_langfuse_lowlevel.value = None
            logger.error(f"{langfuse_e} Langfuse setup failed. Disabling Langfuse.")
    else:
        Settings.callback_manager = None
        global_client_langfuse_lowlevel.value = None
        langfuse_context.configure(
            enabled=False,
        )
        global_settings_langfuse_enabled.value = False
        logger.warning("Not using Langfuse for performance evaluation.")


def update_data_ingestion_settings(callback_data: Any = None):
    """Update the global data ingestion settings based on the user inputs."""
    Settings.chunk_size = global_settings__data_ingestion_chunk_size.value
    Settings.chunk_overlap = global_settings__data_ingestion_chunk_overlap.value


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
                # TODO: Should this be cohere_api_key or api_key?
                cohere_api_key=global_settings__cohere_api_key.value,
                input_type="search_query",
            )
            global_settings__llm_provider_notice.value = """
                Cohere is being used as the language model provider. 
                Ensure that you have set the Cohere API key correctly!
                """
        case constants.LLM_PROVIDER_OPENAI:
            Settings.llm = OpenAI(
                api_key=global_settings__openai_api_key.value,
                model=global_settings__openai_model.value,
                temperature=global_settings__llm_temperature.value,
                system_prompt=global_settings__llm_system_message.value,
            )
            Settings.embed_model = OpenAIEmbedding(
                api_key=global_settings__openai_api_key.value,
            )
            global_settings__llm_provider_notice.value = """
                Open AI is being used as the language model provider. 
                Ensure that you have set the Open AI API key correctly!
                """
        case constants.LLM_PROVIDER_LLAMAFILE:
            Settings.llm = Llamafile(
                base_url=global_settings__llamafile_url.value,
                request_timeout=global_settings__llm_request_timeout.value,
                temperature=global_settings__llm_temperature.value,
                system_prompt=global_settings__llm_system_message.value,
            )
            Settings.embed_model = LlamafileEmbedding(
                base_url=global_settings__llamafile_url.value,
            )
            global_settings__llm_provider_notice.value = constants.EMPTY_STRING
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
            global_settings__llm_provider_notice.value = constants.EMPTY_STRING
    global_settings__embedding_model.value = Settings.embed_model.model_name


def update_chatbot_settings(callback_data: Any = None):
    """Update the chatbot settings."""
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


def update_graph_storage_context(gs: Neo4jPropertyGraphStore = None):
    """Update the graph storage context."""
    if not global_settings__neo4j_disable.value:
        if gs is None:
            gs = Neo4jPropertyGraphStore(
                username=global_settings__neo4j_username.value,
                password=global_settings__neo4j_password.value,
                url=global_settings__neo4j_url.value,
                database=global_settings__neo4j_db_name.value,
            )
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                property_graph_store=gs
            )
        else:
            global_llamaindex_storage_context.value.property_graph_store = gs
    else:
        # Note that the SimplePropertyGraphStore does not support all the features of Neo4j.
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                property_graph_store=SimplePropertyGraphStore()
            )
        else:
            global_llamaindex_storage_context.value.property_graph_store = (
                SimplePropertyGraphStore()
            )


def update_index_documents_vector_storage_context():
    """Update the document and vector storage context."""
    if not global_settings__redis_disable.value:
        global_cache__ingestion.value = RedisCache(
            redis_uri=global_settings__redis_url.value,
        )
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
        vector_store = RedisVectorStore(redis_url=global_settings__redis_url.value)
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                docstore=document_store,
                index_store=index_store,
                vector_stores=[{global_settings__redis_namespace.value, vector_store}],
            )
        else:
            global_llamaindex_storage_context.value.docstore = document_store
            global_llamaindex_storage_context.value.index_store = index_store
            global_llamaindex_storage_context.value.add_vector_store(
                vector_store=vector_store,
                namespace=global_settings__redis_namespace.value,
            )
    else:
        global_cache__ingestion.value = None
        if global_llamaindex_storage_context.value is None:
            global_llamaindex_storage_context.value = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
                vector_stores=[
                    {global_settings__redis_namespace.value, SimpleVectorStore()}
                ],
            )
        else:
            global_llamaindex_storage_context.value.docstore = SimpleDocumentStore()
            global_llamaindex_storage_context.value.index_store = SimpleIndexStore()
            global_llamaindex_storage_context.value.add_vector_store(
                vector_store=SimpleVectorStore(),
                namespace=global_settings__redis_namespace.value,
            )


def initialise_default_settings():
    """Load the global settings from the environment variables."""
    # Load the settings only once.
    if not global_settings_initialised.value:
        """ Load the environment variables from the .env file, if present. """
        load_dotenv()

        """ Language model settings """
        set_global_setting(
            global_settings__language_model_provider,
            constants.ENV_KEY_LLM_PROVIDER,
            constants.DEFAULT_SETTING_LLM_PROVIDER,
        )

        set_global_setting(
            global_settings__cohere_api_key, constants.ENV_KEY_COHERE_API_KEY
        )

        set_global_setting(
            global_settings__cohere_model,
            constants.ENV_KEY_COHERE_MODEL,
            constants.DEFAULT_SETTING_COHERE_MODEL,
        )

        set_global_setting(
            global_settings__openai_api_key, constants.ENV_KEY_OPENAI_API_KEY
        )

        set_global_setting(
            global_settings__openai_model,
            constants.ENV_KEY_OPENAI_MODEL,
            constants.DEFAULT_SETTING_OPENAI_MODEL,
        )

        set_global_setting(
            global_settings__llamafile_url,
            constants.ENV_KEY_LLAMAFILE_URL,
            constants.DEFAULT_SETTING_LLAMAFILE_URL,
        )

        set_global_setting(
            global_settings__ollama_url,
            constants.ENV_KEY_OLLAMA_URL,
            constants.DEFAULT_SETTING_OLLAMA_URL,
        )

        set_global_setting(
            global_settings__ollama_model,
            constants.ENV_KEY_OLLAMA_MODEL,
            constants.DEFAULT_SETTING_OLLAMA_MODEL,
        )

        set_global_setting(
            global_settings__llm_temperature,
            constants.ENV_KEY_LLM_TEMPERATURE,
            constants.DEFAULT_SETTING_LLM_TEMPERATURE,
            float,
        )

        set_global_setting(
            global_settings__llm_request_timeout,
            constants.ENV_KEY_LLM_REQUEST_TIMEOUT,
            constants.DEFAULT_SETTING_LLM_REQUEST_TIMEOUT,
            int,
        )

        set_global_setting(
            global_settings__llm_system_message,
            constants.ENV_KEY_LLM_SYSTEM_MESSAGE,
            constants.DEFAULT_SETTING_LLM_SYSTEM_MESSAGE,
        )

        """ Data ingestion settings """

        set_global_setting(
            global_settings__data_ingestion_chunk_size,
            constants.ENV_KEY_DI_CHUNK_SIZE,
            constants.DEFAULT_SETTING_DI_CHUNK_SIZE,
            int,
        )

        set_global_setting(
            global_settings__data_ingestion_chunk_overlap,
            constants.ENV_KEY_DI_CHUNK_OVERLAP,
            constants.DEFAULT_SETTING_DI_CHUNK_OVERLAP,
            int,
        )

        set_global_setting(
            global_settings__di_enable_title_extractor,
            constants.ENV_KEY_DI_ENABLE_TITLE_EXTRACTOR,
            constants.DEFAULT_SETTING_DI_ENABLE_TITLE_EXTRACTOR,
            bool,
        )
        set_global_setting(
            global_settings__di_enable_title_extractor_nodes,
            constants.ENV_KEY_DI_TITLE_EXTRACTOR_NODES,
            constants.DEFAULT_SETTING_DI_TITLE_EXTRACTOR_NODES,
            int,
        )
        set_global_setting(
            global_settings__di_enable_keyword_extractor,
            constants.ENV_KEY_DI_ENABLE_KEYWORD_EXTRACTOR,
            constants.DEFAULT_SETTING_DI_ENABLE_KEYWORD_EXTRACTOR,
            bool,
        )
        set_global_setting(
            global_settings__di_enable_keyword_extractor_keywords,
            constants.ENV_KEY_DI_KEYWORD_EXTRACTOR_KEYWORDS,
            constants.DEFAULT_SETTING_DI_KEYWORD_EXTRACTOR_KEYWORDS,
            int,
        )
        set_global_setting(
            global_settings__di_enable_qa_extractor,
            constants.ENV_KEY_DI_ENABLE_QA_EXTRACTOR,
            constants.DEFAULT_SETTING_DI_ENABLE_QA_EXTRACTOR,
            bool,
        )
        set_global_setting(
            global_settings__di_enable_qa_extractor_questions,
            constants.ENV_KEY_DI_QA_EXTRACTOR_QUESTIONS,
            constants.DEFAULT_SETTING_DI_QA_EXTRACTOR_QUESTIONS,
            int,
        )
        set_global_setting(
            global_settings__di_enable_summary_extractor,
            constants.ENV_KEY_DI_ENABLE_SUMMARY_EXTRACTOR,
            constants.DEFAULT_SETTING_DI_ENABLE_SUMMARY_EXTRACTOR,
            bool,
        )

        set_global_setting(
            global_settings__di_enable_summary_extractor_summaries,
            constants.ENV_KEY_DI_SUMMARY_EXTRACTOR_SUMMARIES,
            constants.DEFAULT_SETTING_DI_SUMMARY_EXTRACTOR_SUMMARIES,
            str,
            convert_to_list=True,
        )

        """ Index and chat settings """

        set_global_setting(
            global_settings__index_memory_token_limit,
            constants.ENV_KEY_INDEX_MEMORY_TOKEN_LIMIT,
            constants.DEFAULT_SETTING_INDEX_MEMORY_TOKEN_LIMIT,
            int,
        )

        set_global_setting(
            global_settings__index_max_triplets_per_chunk,
            constants.ENV_KEY_INDEX_MAX_TRIPLETS_PER_CHUNK,
            constants.DEFAULT_SETTING_INDEX_MAX_TRIPLETS_PER_CHUNK,
            int,
        )
        set_global_setting(
            global_settings__index_include_embeddings,
            constants.ENV_KEY_INDEX_INCLUDE_EMBEDDINGS,
            constants.DEFAULT_SETTING_INDEX_INCLUDE_EMBEDDINGS,
            bool,
        )

        set_global_setting(
            global_settings__index_chat_mode,
            constants.ENV_KEY_INDEX_CHAT_MODE,
            constants.DEFAULT_SETTING_INDEX_CHAT_MODE,
        )

        """ Neo4j settings """
        set_global_setting(
            global_settings__neo4j_disable,
            constants.ENV_KEY_NEO4J_DISABLE,
            constants.DEFAULT_SETTING_NEO4J_DISABLE,
            bool,
        )
        set_global_setting(
            global_settings__neo4j_url,
            constants.ENV_KEY_NEO4J_URL,
            constants.DEFAULT_SETTING_NEO4J_URL,
        )
        set_global_setting(
            global_settings__neo4j_username,
            constants.ENV_KEY_NEO4J_USERNAME,
            constants.DEFAULT_SETTING_NEO4J_USERNAME,
        )
        set_global_setting(
            global_settings__neo4j_password,
            constants.ENV_KEY_NEO4J_PASSWORD,
        )
        set_global_setting(
            global_settings__neo4j_db_name,
            constants.ENV_KEY_NEO4J_DB_NAME,
            constants.DEFAULT_SETTING_NEO4J_DB_NAME,
        )

        """ Redis settings """
        set_global_setting(
            global_settings__redis_disable,
            constants.ENV_KEY_REDIS_DISABLE,
            constants.DEFAULT_SETTING_REDIS_DISABLE,
            bool,
        )
        set_global_setting(
            global_settings__redis_url,
            constants.ENV_KEY_REDIS_URL,
            constants.DEFAULT_SETTING_REDIS_URL,
        )
        set_global_setting(
            global_settings__redis_namespace,
            constants.ENV_KEY_REDIS_NAMESPACE,
            constants.DEFAULT_SETTING_REDIS_NAMESPACE,
        )

        setup_langfuse()
        # Update all the settings and create objects to be used by other pages and components
        update_llm_settings()
        update_data_ingestion_settings()
        update_chatbot_settings()
        update_graph_storage_context()
        update_index_documents_vector_storage_context()

        # Set this to true so that the settings are not loaded again.
        global_settings_initialised.value = True


# TODO: Not being used and will be removed in the future.
corrective_background_colour: solara.Reactive[str] = solara.reactive(
    constants.EMPTY_STRING
)


def set_theme_colours():
    """Set the theme colours for the Solara app."""

    solara.lab.theme.themes.light.primary = "#2196f3"
    solara.lab.theme.themes.light.secondary = "#ff8159"
    solara.lab.theme.themes.light.accent = "#7abff5"
    solara.lab.theme.themes.light.error = "#f44336"
    solara.lab.theme.themes.light.warning = "#ffc107"
    solara.lab.theme.themes.light.info = "#00bcd4"
    solara.lab.theme.themes.light.success = "#8bc34a"

    solara.lab.theme.themes.dark.primary = "#ff8159"
    solara.lab.theme.themes.dark.secondary = "#673ab7"
    solara.lab.theme.themes.dark.accent = "#c0cf36"
    solara.lab.theme.themes.dark.error = "#f44336"
    solara.lab.theme.themes.dark.warning = "#ffc107"
    solara.lab.theme.themes.dark.info = "#00bcd4"
    solara.lab.theme.themes.dark.success = "#8bc34a"

    # TODO: Not being used and will be removed in the future.
    corrective_background_colour.value = (
        solara.lab.theme.themes.dark.secondary
        if solara.lab.use_dark_effective()
        else solara.lab.theme.themes.light.primary
    )
