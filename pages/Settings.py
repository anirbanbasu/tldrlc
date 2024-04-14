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

"""Module for overall settings of the TLDRLC application."""

import streamlit as st

from utils.streamlit.settings_manager import (
    initialise_shadow_settings,
    load_settings,
    switch_llm_provider,
    update_llm_settings,
    update_graphdb_settings,
    update_document_index_store_settings,
    update_graph_visualisation_settings,
    update_openai_apikey,
    update_cohere_apikey,
)
from utils.streamlit.notifications import write_eu_ai_act_transparency_notice

# Setup page metadata and load settings
if "first_run" not in st.session_state.keys():
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
    load_settings()
else:
    initialise_shadow_settings()

st.title("TLDRLC Settings")

with st.sidebar:
    write_eu_ai_act_transparency_notice()


# print(f"\nTotal session state variables: {len(st.session_state.keys())}")
# for k, v in st.session_state.items():
#     print(f"{k} = {v}")

# Setup the language model settings
with st.expander("**Language model**", expanded=False):
    st.selectbox(
        "LLM provider",
        options=st.session_state.const_llm_providers.list,
        key="shadow__settings_llm__provider",
        help="This setting determines the language model provider. For cloud-based providers, respective API keys are required. For Ollama, the server URL is required.",
        on_change=switch_llm_provider,
    )
    match st.session_state.shadow__settings_llm__provider:
        case st.session_state.const_llm_providers.cohere:
            st.text_input(
                "Cohere API key",
                type="password",
                key="shadow__settings_llm__cohere_api_key",
                help="The API key for Cohere.",
                on_change=update_cohere_apikey,
            )
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_cohere
            )
        case st.session_state.const_llm_providers.openai:
            st.text_input(
                "Open AI API key",
                type="password",
                key="shadow__settings_llm__openai_api_key",
                help="The API key for Open AI.",
                on_change=update_openai_apikey,
            )
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_openai
            )
        case st.session_state.const_llm_providers.ollama:
            st.text_input(
                "Ollama server URL",
                key="shadow__settings_llm__ollama_url",
                help="The URL of the Ollama server. Make sure that the Ollama server is listening on this port of the hostname.",
                on_change=update_llm_settings,
            )
            st.session_state.shadow__settings_llm__model = (
                st.session_state.settings_llm__model_ollama
            )
    if st.session_state.llm_provider_switched:
        update_llm_settings()
    st.text_input(
        "Language model",
        key="shadow__settings_llm__model",
        help="The language model to use with the chosen LLM provider. If using Ollama, the model must be available in the Ollama server. Run `ollama list` to see available models. Or, run `ollama pull <model>` to download a model.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=(
            2.0
            if st.session_state.settings_llm__provider
            == st.session_state.const_llm_providers.openai
            else 1.0
        ),
        step=0.1,
        key="shadow__settings_llm__llm_temperature",
        help="The temperature parameter with a value in $[0,1]$ controls the randomness of the output. A value of $0$ is least random while a value of $1$ will make the output more random, hence more creative.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Chunk size",
        min_value=128,
        max_value=4096,
        step=128,
        key="shadow__settings_llm__llm_chunk_size",
        help="The chunk size is the maximum number of tokens in each chunk of text. The language model will process the text in chunks of this size.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Chunk overlap",
        min_value=16,
        max_value=128,
        step=16,
        key="shadow__settings_llm__llm_chunk_overlap",
        help="The chunk overlap is the number of tokens that overlap between adjacent chunks. This helps the language model to generate coherent text across chunks.",
        on_change=update_llm_settings,
    )
    st.text_area(
        "System message",
        key="shadow__settings_llm__llm_system_message",
        help="The system message is an optional initial prompt for the language model. It should be a short sentence or a few words that describe the context of the conversation. The language model will use this prompt to generate a response.",
        on_change=update_llm_settings,
    )

# Setup the language model chat index settings
with st.expander("**Language model chat index**", expanded=False):
    st.slider(
        "Memory token limit",
        min_value=2048,
        max_value=16384,
        step=512,
        key="shadow__settings_llm__index_memory_token_limit",
        help="The maximum number of tokens to store in the chat memory buffer.",
        on_change=update_llm_settings,
    )
    st.slider(
        "Max triplets per chunk",
        min_value=1,
        max_value=32,
        step=1,
        key="shadow__settings_llm__index_max_triplets_per_chunk",
        help="The maximum number of knowledge graph triplets to extract from each chunk of text.",
        on_change=update_llm_settings,
    )
    st.selectbox(
        "Chat mode",
        options=["condense_plus_context", "context"],
        disabled=True,
        key="shadow__settings_llm__index_chat_mode",
        help="The chat mode determines how the knowledge graph is used to generate responses. The `condense_plus_context` mode condenses the question in the prompt first and then uses the knowledge graph to provide context. The `context` mode uses the knowledge graph to provide context for the language model but does not condense the input prompt.",
        on_change=update_llm_settings,
    )
    st.checkbox(
        "Include embeddings in index",
        key="shadow__settings_llm__index_include_embeddings",
        help="If checked, the knowledge graph index will include embeddings for each document.",
        on_change=update_llm_settings,
    )

# Setup the graph database settings
with st.expander("**Graph storage**", expanded=False):
    st.checkbox(
        "Do not use graph database",
        key="shadow__settings_graphdb__disable",
        help="If checked, the knowledge graph will reside entirely in memory.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Neo4j URL",
        disabled=st.session_state.settings_graphdb__disable,
        key="shadow__settings_graphdb__url",
        help="The URL of the on-premises Neo4j database or the cloud-hosted Neo4j Aura DB.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Username",
        disabled=st.session_state.settings_graphdb__disable,
        key="shadow__settings_graphdb__username",
        help="The username to connect to the Neo4j database.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Password",
        disabled=st.session_state.settings_graphdb__disable,
        key="shadow__settings_graphdb__password",
        type="password",
        help="The password to connect to the Neo4j database.",
        on_change=update_graphdb_settings,
    )
    st.text_input(
        label="Database name",
        disabled=st.session_state.settings_graphdb__disable,
        key="shadow__settings_graphdb__dbname",
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
with st.expander("**Document and index storage**", expanded=False):
    st.checkbox(
        "Do not use Redis",
        key="shadow__settings_redis__disable",
        help="If checked, the index and documents will reside entirely in memory.",
        on_change=update_document_index_store_settings,
    )
    st.text_input(
        label="Redis connection URL",
        disabled=st.session_state.settings_redis__disable,
        key="shadow__settings_redis__url",
        help="The URL of the Redis key-value storage.",
        type="password",
        on_change=update_document_index_store_settings,
    )
    st.text_input(
        label="Redis namespace",
        disabled=st.session_state.settings_redis__disable,
        key="shadow__settings_redis__namespace",
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
with st.expander("**Graph visualisation**", expanded=False):
    st.slider(
        "Max nodes",
        min_value=25,
        max_value=100,
        step=1,
        key="shadow__settings_kgvis__max_nodes",
        help="The maximum number of nodes, starting with the most connected ones, to display in the visualisation.",
    )
    st.slider(
        "Height (in pixels)",
        min_value=250,
        max_value=2500,
        step=50,
        key="shadow__settings_kgvis__height",
        help="The height, in pixels, of the knowledge graph rendering.",
        on_change=update_graph_visualisation_settings,
    )
    st.selectbox(
        "Graph layout",
        options=["circular", "planar", "shell", "spectral", "spring"],
        key="shadow__settings_kgvis__layout",
        help="The visual layout mode of the knowledge graph.",
    )
    st.checkbox(
        "Physics enabled",
        key="shadow__settings_kgvis__physics_enabled",
        help="If checked, the physics simulation will be enabled for the knowledge graph rendering.",
    )
