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

LIST_OF_SUPPORTED_LLM_PROVIDERS = ["Cohere", "Ollama", "Open AI"]
SOURCE_TYPE_WIKIPEDIA = "Wikipedia"
SOURCE_TYPE_ARXIV = "arXiv"
SOURCE_TYPE_PUBMED = "Pubmed"
SOURCE_TYPE_WEBPAGE = "Webpage"
SOURCE_TYPE_PDF = "PDF"
LIST_OF_SOURCE_TYPES = [
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

# Settings
DEFAULT_SETTING_LLM_PROVIDER = "Ollama"
DEFAULT_SETTING_OLLAMA_URL = "http://localhost:11434"
DEFAULT_SETTING_OLLAMA_MODEL = "mistral"
DEFAULT_SETTING_COHERE_MODEL = "command"
DEFAULT_SETTING_OPENAI_MODEL = "gpt-3.5-turbo"

DEFAULT_SETTING_LLM_REQUEST_TIMEOUT = 120
DEFAULT_SETTING_LLM_TEMPERATURE = 0.0
DEFAULT_SETTING_LLM_SYSTEM_MESSAGE = "You are an intelligent assistant. You respond to questions about articles from various sources such as Wikipedia, arXiv, Pubmed and so on. You generate your answers based ONLY on information in those articles that are provided to you as context information. If you are unsure about an answer or if the user query cannot be answered using information in the context then say that you do not know. If the user ask you to stop, immediately stop answering."
DEFAULT_SETTING_LLM_CHUNK_SIZE = 1024
DEFAULT_SETTING_LLM_CHUNK_OVERLAP = 64

DEFAULT_SETTING_INDEX_MEMORY_TOKEN_LIMIT = 4096
DEFAULT_SETTING_INDEX_MAX_TRIPLETS_PER_CHUNK = 8
DEFAULT_SETTING_INDEX_INCLUDE_EMBEDDINGS = True
DEFAULT_SETTING_INDEX_CHAT_MODE = "context"

DEFAULT_SETTING_NEO4J_DISABLE = False
DEFAULT_SETTING_NEO4J_URL = "bolt://localhost:7687"
DEFAULT_SETTING_NEO4J_USERNAME = "neo4j"
DEFAULT_SETTING_NEO4J_DB_NAME = "neo4j"

DEFAULT_SETTING_REDIS_DISABLE = False
DEFAULT_SETTING_REDIS_URL = "redis://localhost:6379"
DEFAULT_SETTING_REDIS_NAMESPACE = "tldrlc"

DEFAULT_SETTING_KG_VIS_HEIGHT = 800
DEFAULT_SETTING_KG_VIS_MAX_NODES = 100
DEFAULT_SETTING_KG_VIS_MAX_DEPTH = 3
DEFAULT_SETTING_KG_VIS_LAYOUT = "spring"
DEFAULT_SETTING_KG_VIS_PHYSICS_ENABLED = True

DEFAULT_SETTING_EVAL_USE_LANGFUSE = False

# Chat message related keys
CHAT_KEY_ROLE = "role"
CHAT_KEY_CONTENT = "content"
CHAT_KEY_TIMESTAMP = "timestamp"
CHAT_KEY_VALUE_USER = "user"
CHAT_KEY_VALUE_ASSISTANT = "assistant"

# UI related parameters
UI_STATUS_CONTAINER_HEIGHT = 300
UI_CHAT_CONTAINER_HEIGHT = 400
