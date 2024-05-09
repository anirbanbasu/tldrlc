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

"""Various constants used in the project."""

EMPTY_STRING = ""
ISO639SET1_LANGUAGE_ENGLISH = "en"

CHAR_ENCODING_UTF8 = "utf-8"

# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
LOG_FORMAT = "%(levelname)s:\t%(message)s (%(module)s/%(filename)s::%(funcName)s:L%(lineno)d@%(asctime)s) [P%(process)d:T%(thread)d]"

LLM_PROVIDER_COHERE = "Cohere"
LLM_PROVIDER_OLLAMA = "Ollama"
LLM_PROVIDER_OPENAI = "Open AI"
LIST_OF_SUPPORTED_LLM_PROVIDERS = [
    LLM_PROVIDER_COHERE,
    LLM_PROVIDER_OLLAMA,
    LLM_PROVIDER_OPENAI,
]

INDEX_CHAT_MODE_CONTEXT = "context"
INDEX_CHAT_MODE_CONDENSE_PLUS_CONTEXT = "condense_plus_context"
INDEX_CHAT_MODE_REACT = "react"
LIST_OF_INDEX_CHAT_MODES = [
    INDEX_CHAT_MODE_CONTEXT,
    INDEX_CHAT_MODE_CONDENSE_PLUS_CONTEXT,
    INDEX_CHAT_MODE_REACT,
]
GRAPH_VIS_LAYOUT_SPRING = "spring"
GRAPH_VIS_LAYOUT_PLANAR = "planar"
GRAPH_VIS_LAYOUT_CIRCULAR = "circular"
GRAPH_VIS_LAYOUT_SPECTRAL = "spectral"
GRAPH_VIS_LAYOUT_SHELL = "shell"
LIST_OF_GRAPH_VIS_LAYOUTS = [
    GRAPH_VIS_LAYOUT_CIRCULAR,
    GRAPH_VIS_LAYOUT_PLANAR,
    GRAPH_VIS_LAYOUT_SHELL,
    GRAPH_VIS_LAYOUT_SPECTRAL,
    GRAPH_VIS_LAYOUT_SPRING,
]

SOURCE_TYPE_WIKIPEDIA = "Wikipedia"
SOURCE_TYPE_ARXIV = "arXiv"
SOURCE_TYPE_PUBMED = "Pubmed"
SOURCE_TYPE_WEBPAGE = "Webpage"
SOURCE_TYPE_PDF = "PDF"
SOURCE_TYPE_INDICES = "existing indices"
LIST_OF_SOURCE_TYPES = [
    SOURCE_TYPE_INDICES,
    SOURCE_TYPE_WIKIPEDIA,
    SOURCE_TYPE_ARXIV,
    SOURCE_TYPE_PUBMED,
    SOURCE_TYPE_WEBPAGE,
    SOURCE_TYPE_PDF,
]
LIST_OF_SOURCE_TYPES_DESCRIPTIONS = [
    "Wikipedia article",
    "arXiv query",
    "Pubmed query",
    "URL of a webpage",
    "URL of a PDF file",
]
WEBPAGE_READER_BEAUTIFUL_SOUP = "BeautifulSoup"
WEBPAGE_READER_TRAFILATURA = "Trafilatura"

LIST_OF_WEBPAGE_READERS = [
    WEBPAGE_READER_BEAUTIFUL_SOUP,
    WEBPAGE_READER_TRAFILATURA,
]

# Environment variables
ENV_KEY_LLM_PROVIDER = "LLM_PROVIDER"

ENV_KEY_COHERE_API_KEY = "COHERE_API_KEY"
ENV_KEY_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_KEY_OLLAMA_URL = "OLLAMA_URL"
ENV_KEY_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_KEY_COHERE_MODEL = "COHERE_MODEL"
ENV_KEY_OPENAI_MODEL = "OPENAI_MODEL"

ENV_KEY_LLM_REQUEST_TIMEOUT = "LLM_REQUEST_TIMEOUT"
ENV_KEY_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_KEY_LLM_CHUNK_SIZE = "LLM_CHUNK_SIZE"
ENV_KEY_LLM_CHUNK_OVERLAP = "LLM_CHUNK_OVERLAP"
ENV_KEY_LLM_SYSTEM_MESSAGE = "LLM_SYSTEM_MESSAGE"

ENV_KEY_INDEX_MEMORY_TOKEN_LIMIT = "INDEX_MEMORY_TOKEN_LIMIT"
ENV_KEY_INDEX_MAX_TRIPLETS_PER_CHUNK = "INDEX_MAX_TRIPLETS_PER_CHUNK"
ENV_KEY_INDEX_INCLUDE_EMBEDDINGS = "INDEX_INCLUDE_EMBEDDINGS"
ENV_KEY_INDEX_CHAT_MODE = "INDEX_CHAT_MODE"

ENV_KEY_NEO4J_DISABLE = "NEO4J_DISABLE"
ENV_KEY_NEO4J_URL = "NEO4J_URL"
ENV_KEY_NEO4J_USERNAME = "NEO4J_USERNAME"
ENV_KEY_NEO4J_PASSWORD = "NEO4J_PASSWORD"
ENV_KEY_NEO4J_DB_NAME = "NEO4J_DB_NAME"

ENV_KEY_REDIS_DISABLE = "REDIS_DISABLE"
ENV_KEY_REDIS_URL = "REDIS_URL"
ENV_KEY_REDIS_NAMESPACE = "REDIS_NAMESPACE"

ENV_KEY_KG_VIS_HEIGHT = "KG_VIS_HEIGHT"
ENV_KEY_KG_VIS_MAX_NODES = "KG_VIS_MAX_NODES"
ENV_KEY_KG_VIS_MAX_DEPTH = "KG_VIS_MAX_DEPTH"
ENV_KEY_KG_VIS_LAYOUT = "KG_VIS_LAYOUT"
ENV_KEY_KG_VIS_PHYSICS_ENABLED = "KG_VIS_PHYSICS_ENABLED"

ENV_KEY_EVAL_USE_LANGFUSE = "EVAL_USE_LANGFUSE"
ENV_KEY_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_KEY_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_KEY_LANGFUSE_HOST = "LANGFUSE_HOST"
ENV_KEY_LANGFUSE_TRACE_TAGS = "LANGFUSE_TRACE_TAGS"

ENV_KEY_LOG_LEVEL = "LOG_LEVEL"

# Settings
DEFAULT_SETTING_LLM_PROVIDER = "Ollama"
DEFAULT_SETTING_OLLAMA_URL = "http://localhost:11434"
DEFAULT_SETTING_OLLAMA_MODEL = "llama3"
DEFAULT_SETTING_COHERE_MODEL = "command-r-plus"
DEFAULT_SETTING_OPENAI_MODEL = "gpt-3.5-turbo-0125"

DEFAULT_SETTING_LLM_REQUEST_TIMEOUT = "120"
DEFAULT_SETTING_LLM_TEMPERATURE = "0.0"
DEFAULT_SETTING_LLM_SYSTEM_MESSAGE = "You are an intelligent assistant. You provide concise and informative responses to questions from the user using only the information given to you as context. If you are unsure about an answer or if the user question cannot be answered using information in the context then say that you do not know. Cite the actual text from the context as a reference with your answer."
DEFAULT_SETTING_LLM_CHUNK_SIZE = "1024"
DEFAULT_SETTING_LLM_CHUNK_OVERLAP = "64"

DEFAULT_SETTING_INDEX_MEMORY_TOKEN_LIMIT = "4096"
DEFAULT_SETTING_INDEX_MAX_TRIPLETS_PER_CHUNK = "8"
DEFAULT_SETTING_INDEX_INCLUDE_EMBEDDINGS = "True"
DEFAULT_SETTING_INDEX_CHAT_MODE = "context"

DEFAULT_SETTING_NEO4J_DISABLE = "False"
DEFAULT_SETTING_NEO4J_URL = "bolt://localhost:7687"
DEFAULT_SETTING_NEO4J_USERNAME = "neo4j"
DEFAULT_SETTING_NEO4J_DB_NAME = "neo4j"

DEFAULT_SETTING_REDIS_DISABLE = "False"
DEFAULT_SETTING_REDIS_URL = "redis://localhost:6379"
DEFAULT_SETTING_REDIS_NAMESPACE = "tldrlc"

DEFAULT_SETTING_KG_VIS_HEIGHT = "800"
DEFAULT_SETTING_KG_VIS_MAX_NODES = "100"
DEFAULT_SETTING_KG_VIS_MAX_DEPTH = "3"
DEFAULT_SETTING_KG_VIS_LAYOUT = "spring"
DEFAULT_SETTING_KG_VIS_PHYSICS_ENABLED = "True"

DEFAULT_SETTING_EVAL_USE_LANGFUSE = "False"

DEFAULT_SETTING_LOG_LEVEL = "30"

# Chat message related keys
CHAT_KEY_ROLE = "role"
CHAT_KEY_CONTENT = "content"
CHAT_KEY_TIMESTAMP = "timestamp"
CHAT_KEY_LLM_PROVIDER = "llm_provider"
CHAT_KEY_LLM_MODEL_NAME = "llm_model_name"
CHAT_KEY_VALUE_USER = "user"
CHAT_KEY_VALUE_ASSISTANT = "assistant"

# Data source related keys
DEFAULT_DATA_SOURCE_WIKIPEDIA_LANGUAGE_PREFIX = "en"

# UI related parameters
UI_STATUS_CONTAINER_HEIGHT = "300"
UI_CHAT_CONTAINER_HEIGHT = "400"

UI_SOLARA_NOTICE_REMOVE = """
        .v-application--wrap > div:nth-child(2) > div:nth-child(2){
            display: none !important;
        }
        """

NOTICE_EU_AI_ACT__MESSAGE = """
            By using this app, you are interacting with an artificial intelligence (AI) system. 
            **You are advised not to take any of its responses as facts**. The AI system is not a 
            substitute for professional advice. If you are unsure about any information, please 
            consult a professional in the field.
            """
NOTICE_EU_AI_ACT__TITLE = "EU AI Act Transparency notice"
NOTICE_EU_AI_ACT__OK = "Okay, understood."
NOTICE_EU_AI_ACT__CANCEL = "Hide this button."
