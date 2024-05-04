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

"""Solara web module for managing settings of the application."""

import os
import logging
from typing import Any
from urllib.parse import urlparse

from redis import Redis
import solara
import solara.lab
from solara.alias import rv
import solara.tasks

import utils.constants as constants
import utils.global_state as global_state

from llama_index.graph_stores.neo4j import Neo4jGraphStore

logger = logging.getLogger(__name__)
# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.basicConfig(
    # See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
    format=constants.LOG_FORMAT,
    level=int(os.getenv(constants.ENV_KEY_LOG_LEVEL)),
    encoding=constants.CHAR_ENCODING_UTF8,
)


@solara.component
def LLMSettingsComponent():
    """Component for the language model settings."""

    with solara.Card(
        title="Language model provider and language model",
        subtitle="""
            The language model is the core of the chatbot. It is responsible for responding to the user's inputs.
            """,
        elevation=0,
    ):
        solara.Select(
            label="Language model provider",
            values=constants.LIST_OF_SUPPORTED_LLM_PROVIDERS,
            value=global_state.global_settings__language_model_provider,
            on_value=global_state.update_llm_settings,
        )
        match global_state.global_settings__language_model_provider.value:
            case constants.LLM_PROVIDER_COHERE:
                solara.InputText(
                    label="Cohere API key",
                    value=global_state.global_settings__cohere_api_key,
                    password=True,
                    message="You can get an API key from the Cohere website.",
                    on_value=global_state.update_cohere_apikey,
                )
                solara.InputText(
                    label="Cohere model",
                    value=global_state.global_settings__cohere_model,
                    on_value=global_state.update_llm_settings,
                )
            case constants.LLM_PROVIDER_OPENAI:
                solara.InputText(
                    label="Open AI API key",
                    value=global_state.global_settings__openai_api_key,
                    password=True,
                    message="You can get an API key from the Open AI website.",
                    on_value=global_state.update_openai_apikey,
                )
                solara.InputText(
                    label="Open AI model",
                    value=global_state.global_settings__openai_model,
                    on_value=global_state.update_llm_settings,
                )
            case constants.LLM_PROVIDER_OLLAMA:
                solara.InputText(
                    label="Ollama URL",
                    value=global_state.global_settings__ollama_url,
                    message="The URL must point to a running Ollama server.",
                    on_value=global_state.update_llm_settings,
                )
                solara.InputText(
                    label="Ollama model",
                    value=global_state.global_settings__ollama_model,
                    message="The model must be available on the selected Ollama server.",
                    on_value=global_state.update_llm_settings,
                )
        solara.SliderInt(
            label="Chunk size",
            min=128,
            max=4096,
            step=64,
            value=global_state.global_settings__llm_chunk_size,
            tick_labels="end_points",
            on_value=global_state.update_llm_settings,
        )
        solara.SliderInt(
            label="Chunk overlap",
            min=16,
            max=128,
            step=8,
            value=global_state.global_settings__llm_chunk_overlap,
            tick_labels="end_points",
            on_value=global_state.update_llm_settings,
        )
        solara.SliderFloat(
            label="Temperature",
            min=0.0,
            max=(
                2.0
                if global_state.global_settings__language_model_provider.value
                == constants.LLM_PROVIDER_OPENAI
                else 1.0
            ),
            step=0.1,
            value=global_state.global_settings__llm_temperature,
            tick_labels="end_points",
            on_value=global_state.update_llm_settings,
        )
        solara.SliderInt(
            label="LLM request timeout, in seconds",
            min=60,
            max=600,
            step=30,
            value=global_state.global_settings__llm_request_timeout,
            tick_labels="end_points",
            on_value=global_state.update_llm_settings,
        )

        rv.Textarea(
            label="System message",
            v_model=global_state.global_settings__llm_system_message.value,
            on_v_model=global_state.global_settings__llm_system_message.set,
            rows=2,
        )
    # update_llm_settings()


@solara.component
def ChatbotSettingsComponent():
    """Component for the chatbot settings."""

    with solara.Card(
        title="Chat and indexing",
        subtitle="""
            The chatbot settings control how the chatbot interacts with the user.
            """,
        elevation=0,
    ):
        solara.SliderInt(
            label="Memory token limit",
            min=2048,
            max=16384,
            step=1024,
            value=global_state.global_settings__index_memory_token_limit,
            on_value=global_state.update_chatbot_settings,
            tick_labels="end_points",
        )
        solara.SliderInt(
            label="Max triplets per chunk",
            min=1,
            max=16,
            step=1,
            value=global_state.global_settings__index_max_triplets_per_chunk,
            tick_labels="end_points",
        )
        solara.Checkbox(
            label="Include embeddings",
            value=global_state.global_settings__index_include_embeddings,
        )
        solara.Select(
            label="Chat mode",
            values=constants.LIST_OF_INDEX_CHAT_MODES,
            value=global_state.global_settings__index_chat_mode,
            # disabled=True,
        )


@solara.component
def GraphDBSettingsComponent():
    """Component for the graph database settings."""
    status: solara.Reactive[Any] = solara.use_reactive(None)

    def test_graphdb_connection(callback_data: Any = None):
        """Test the graph database connection."""
        nonlocal status
        if not global_state.global_settings__neo4j_disable.value:
            try:
                gs = Neo4jGraphStore(
                    username=global_state.global_settings__neo4j_username.value,
                    password=global_state.global_settings__neo4j_password.value,
                    url=global_state.global_settings__neo4j_url.value,
                    database=global_state.global_settings__neo4j_db_name.value,
                )
                status.value = solara.Success(
                    f"Connected to the graph database at {global_state.global_settings__neo4j_url.value}."
                )
                global_state.update_graph_storage_context(gs)
            except Exception as e:
                status.value = solara.Error(f"{e}")
        else:
            status.value = solara.Warning(
                "Graph database connection has been disabled. Using in-memory storage."
            )
            global_state.update_graph_storage_context()

    with solara.Card(
        title="Graph storage",
        subtitle="""
            The graph storage settings control how the application stores the knowledge graph. 
            Neo4j is used to store the knowledge graph. If Neo4j is not available, the chatbot
            will store the knowledge graph in memory, which will be lost when the browser session expires.
            """,
        elevation=0,
    ):
        solara.Checkbox(
            label="Disable Neo4j and use in-memory storage",
            value=global_state.global_settings__neo4j_disable,
            on_value=test_graphdb_connection,
        )
        solara.InputText(
            label="Neo4j URL",
            value=global_state.global_settings__neo4j_url,
            message="The URL must point to a running Neo4j server.",
            disabled=global_state.global_settings__neo4j_disable.value,
            on_value=test_graphdb_connection,
        )
        solara.InputText(
            label="Neo4j username",
            value=global_state.global_settings__neo4j_username,
            disabled=global_state.global_settings__neo4j_disable.value,
            on_value=test_graphdb_connection,
        )
        solara.InputText(
            label="Neo4j password",
            value=global_state.global_settings__neo4j_password,
            password=True,
            disabled=global_state.global_settings__neo4j_disable.value,
            on_value=test_graphdb_connection,
        )
        solara.InputText(
            label="Neo4j database",
            value=global_state.global_settings__neo4j_db_name,
            disabled=global_state.global_settings__neo4j_disable.value,
            on_value=test_graphdb_connection,
        )

    if status.value is not None:
        solara.display(status.value)

    # update_graph_storage_context()


@solara.component
def DocumentsIndexStorageSettingsComponent():
    """Component for the documents and index storage settings."""
    status: solara.Reactive[Any] = solara.use_reactive(None)

    def test_redis_connection(callback_data: Any = None):
        """Test the Redis connection."""
        nonlocal status
        if not global_state.global_settings__redis_disable.value:
            try:
                parsed_url = urlparse(global_state.global_settings__redis_url.value)
                redis = Redis(
                    host=parsed_url.hostname,
                    port=parsed_url.port,
                    db=0,
                    decode_responses=True,
                    ssl=True if parsed_url.scheme == "rediss" else False,
                    username=parsed_url.username if parsed_url.username else None,
                    password=parsed_url.password if parsed_url.password else None,
                )
                redis.ping()
                redis.close()
                status.value = solara.Success(
                    f"Connected to the Redis server at {global_state.global_settings__redis_url.value}."
                )
            except Exception as e:
                status.value = solara.Error(f"{e}")
        else:
            status.value = solara.Warning(
                "Redis connection has been disabled. Using in-memory storage."
            )
        global_state.update_index_documents_storage_context()

    with solara.Card(
        title="Documents and index storage",
        subtitle="""
            The documents and index storage settings control how the chatbot stores the 
            documents and indexes. Redis is used to store the documents and indexes. 
            If Redis is not available, the chatbot will store the documents and indexes 
            in memory, which will be lost when the browser session expires.
            """,
        elevation=0,
    ):
        solara.Checkbox(
            label="Disable Redis",
            value=global_state.global_settings__redis_disable,
            on_value=test_redis_connection,
        )
        solara.InputText(
            label="Redis URL",
            value=global_state.global_settings__redis_url,
            message="The URL must point to a running Redis server.",
            disabled=global_state.global_settings__redis_disable.value,
            password=(
                True
                if urlparse(global_state.global_settings__redis_url.value).password
                is not None
                else False
            ),
            on_value=test_redis_connection,
        )
        solara.InputText(
            label="Redis namespace",
            value=global_state.global_settings__redis_namespace,
            disabled=global_state.global_settings__redis_disable.value,
            on_value=test_redis_connection,
        )

    if status.value is not None:
        solara.display(status.value)

    # update_index_documents_storage_context()


@solara.component
def GraphVisualisationSettingsComponent():
    """Component for the graph visualisation settings."""
    with solara.Card(
        title="Graph visualisation",
        subtitle="""
            The graph visualisation settings control how the knowledge graph is visualised.
        """,
        elevation=0,
    ):
        solara.SliderInt(
            label="Height (in pixels)",
            min=500,
            max=2500,
            step=250,
            value=global_state.global_settings__kg_vis_height,
            tick_labels="end_points",
        )
        solara.SliderInt(
            label="Max nodes",
            min=50,
            max=1000,
            step=25,
            value=global_state.global_settings__kg_vis_max_nodes,
            tick_labels="end_points",
        )
        solara.SliderInt(
            label="Max depth",
            min=1,
            max=10,
            step=1,
            value=global_state.global_settings__kg_vis_max_depth,
            tick_labels="end_points",
        )
        solara.Select(
            label="Layout",
            values=constants.LIST_OF_GRAPH_VIS_LAYOUTS,
            value=global_state.global_settings__kg_vis_layout,
        )


@solara.component
def Page():
    """Main settings page."""
    # Remove the "This website runs on Solara" message
    solara.Style(constants.UI_SOLARA_NOTICE_REMOVE)

    global_state.initialise_default_settings()

    with solara.Head():
        solara.Title("Settings")

    with solara.AppBarTitle():
        solara.Markdown("# Settings", style={"color": "#FFFFFF"})

    with solara.AppBar():
        solara.lab.ThemeToggle()

    with solara.lab.Tabs(vertical=True, grow=False):
        with solara.lab.Tab("Language model", icon_name="mdi-face-agent"):
            LLMSettingsComponent()
        with solara.lab.Tab("Chatbot index", icon_name="mdi-book-open"):
            ChatbotSettingsComponent()
        with solara.lab.Tab("Graph storage", icon_name="mdi-graph-outline"):
            GraphDBSettingsComponent()
        with solara.lab.Tab(
            "Documents and index storage", icon_name="mdi-file-document-outline"
        ):
            DocumentsIndexStorageSettingsComponent()
        with solara.lab.Tab("Graph visualisation", icon_name="mdi-image-filter-none"):
            GraphVisualisationSettingsComponent()
