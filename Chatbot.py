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

"""Module for the main page of the Streamlit app for TLDRLC."""

import datetime
import hashlib
import json
import os
import time
import requests
from pyvis.network import Network
import networkx as nx
import wikipedia
from llama_index.core import KnowledgeGraphIndex
from llama_index.readers.web import BeautifulSoupWebReader, TrafilaturaWebReader
from llama_index.readers.papers import ArxivReader, PubmedReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

import streamlit as st

from utils.constants import (
    CHAT_KEY_CONTENT,
    CHAT_KEY_ROLE,
    CHAT_KEY_TIMESTAMP,
    CHAT_KEY_VALUE_ASSISTANT,
    CHAT_KEY_VALUE_USER,
    LIST_OF_SOURCE_TYPES,
    LIST_OF_SOURCE_TYPES_DESCRIPTIONS,
    SOURCE_TYPE_ARXIV,
    SOURCE_TYPE_PDF,
    SOURCE_TYPE_PUBMED,
    SOURCE_TYPE_WEBPAGE,
    SOURCE_TYPE_WIKIPEDIA,
    UI_CHAT_CONTAINER_HEIGHT,
    UI_STATUS_CONTAINER_HEIGHT,
    WEBPAGE_READER_BEAUTIFUL_SOUP,
    WEBPAGE_READER_TRAFILATURA,
)
from utils.streamlit.settings_manager import (
    copy_shadow_ui_widget_session_keys,
    initialise_shadow_settings,
    load_settings,
    require_force_index_rebuild,
)
from utils.streamlit.notifications import write_eu_ai_act_transparency_notice

# Setup page metadata and load settings
# Fun emojis for messages: https://github.com/ikatyang/emoji-cheat-sheet
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

st.title("TLDRLC: Too Long, Didn't Read, Let's Chat!")

# Add the EU AI Act Article 52 transparency notice to the side bar
with st.sidebar:
    write_eu_ai_act_transparency_notice()

st.markdown(
    """
    This is an experimental chatbot that lets you chat about document sources using a Knowledge Graph (KG) 
    based Retrieval Augmented Generation (RAG).
    """
)


def stream_wrapper(streaming_response):
    """Wrapper for the streaming response from the chat engine."""
    # Do not use yield from as suggested by Pylint. Irrelevant for ruff.
    for token in streaming_response.response_gen:
        # Filter out symbols that break formatting
        if token == "$":
            # Escape the $ sign, otherwise LaTeX formatting will be triggered!
            yield f"\\${token}"
        else:
            yield token


def save_graph_visualisation(_kgindex: KnowledgeGraphIndex) -> str:
    """Save the graph visualisation to a HTML file."""
    # Try to make a graph from the index at all depths, but limit the number of nodes
    graph = nx.Graph()
    already_added_nodes = set()
    nodes = list(_kgindex.index_struct.table.keys())[
        slice(st.session_state.settings_kgvis__max_nodes)
    ]

    for node in nodes:
        triplets = _kgindex.storage_context.graph_store.get(node)
        if node not in already_added_nodes:
            # This node title reformatting is purely for visualisation purposes
            node_title = node.replace("_", " ").capitalize()
            graph.add_node(
                node,
                label=node_title,
                title=node_title,
                # Value signifies the number of outgoing edges from this node
                value=len(triplets) + 1,
            )
            already_added_nodes.add(node)
        for triplet in triplets:
            if triplet[1] in nodes:
                # This edge title reformatting is purely for visualisation purposes
                edge_title = triplet[0].replace("_", " ").capitalize()
                graph.add_edge(node, triplet[1], title=edge_title)
    # Use user-specified layout for the graph
    _scale_factor = len(nodes) * 24
    match st.session_state.settings_kgvis__layout:
        case "circular":
            positions = nx.circular_layout(graph, scale=_scale_factor)
        case "planar":
            positions = nx.planar_layout(graph, scale=_scale_factor)
        case "shell":
            positions = nx.shell_layout(graph, scale=_scale_factor)
        case "spectral":
            positions = nx.spectral_layout(graph, scale=_scale_factor)
        case "spring":
            positions = nx.spring_layout(graph, scale=_scale_factor)
        case _:
            positions = nx.spring_layout(graph, scale=_scale_factor)

    network = Network(
        notebook=False,
        height=f"{st.session_state.settings_kgvis__height}px",
        width="100%",
        cdn_resources="remote",
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
    st.session_state.data__kg_html = network.generate_html()


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
        st.warning(":bathtub: Perhaps overwriting, existing knowledge graphs...")
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
        f":snail: Building index using {st.session_state.settings_llm__model} model of {st.session_state.settings_llm__provider}. This WILL take time, unless it is read from the cache. Sit back and relax!"
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
    if (
        st.session_state.settings_llm__provider
        == st.session_state.const_llm_providers.cohere
    ):
        post_processors = [
            CohereRerank(api_key=st.session_state.settings_llm__cohere_api_key)
        ]
    else:
        post_processors = [SentenceTransformerRerank()]
    if "index" in st.session_state and st.session_state.index is not None:
        if "memory" in st.session_state:
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
    if "memory" in st.session_state:
        st.session_state.memory.reset()
    st.session_state.messages.clear()


def export_chat_history_as_json():
    """Export the chat history to a file."""
    if len(st.session_state.messages) > 0:
        with container_data_source.expander(
            f"**Exported chat history of {len(st.session_state.messages)} messages**",
            expanded=True,
        ):
            with st.container(height=UI_STATUS_CONTAINER_HEIGHT, border=0):
                st.json(
                    json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
                )


# Main page content: col1_main on LHS, col2_main on RHS
col1_main, col2_main = st.columns(2, gap="large")

container_data_source = col1_main.container()
container_data_source.markdown("## Data source")

container_data_source.radio(
    "Source type",
    LIST_OF_SOURCE_TYPES,
    captions=LIST_OF_SOURCE_TYPES_DESCRIPTIONS,
    help="Select the type of source document to fetch. For arXiv and Pubmed, a search query may result in many relevant articles. Only the top 10 will be used as input sources.",
    key="shadow__ui__radio_source_type",
    horizontal=True,
    on_change=require_force_index_rebuild,
)


def format_wikipedia_languages(option):
    """Format the Wikipedia language options in a `{prefix}: {language name}` format."""
    return f"{option}: {wikipedia.languages()[option]}"


col1_wikipedia_prefix, col2_webpage_reader = container_data_source.columns(2)

col1_wikipedia_prefix.selectbox(
    "Wikipedia language prefix",
    options=list(wikipedia.languages().keys()),
    format_func=format_wikipedia_languages,
    key="shadow__ui__selectbox_wikipedia_prefix",
    disabled=st.session_state.shadow__ui__radio_source_type != SOURCE_TYPE_WIKIPEDIA,
    help="Select the Wikipedia language prefix (defaults to `en: English`). _Only available for selection if you use `Wikipedia` as the input source._ Check supported languages here: [List of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).",
    on_change=copy_shadow_ui_widget_session_keys,
)

col2_webpage_reader.selectbox(
    "Web page reader",
    key="shadow__ui__select_webpage_reader",
    options=[WEBPAGE_READER_BEAUTIFUL_SOUP, WEBPAGE_READER_TRAFILATURA],
    disabled=st.session_state.shadow__ui__radio_source_type != SOURCE_TYPE_WEBPAGE,
    help="Select the type of reader to use to extract text from the web page. _Only available for selection if you use `Web page` as the input source._ One reader maybe more efficient than another in the task of extracting text, depending on the source web page.",
    on_change=copy_shadow_ui_widget_session_keys,
)

container_data_source.text_input(
    "Source documents or existing index ID",
    key="shadow__ui__txtinput_document_source",
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
        copy_shadow_ui_widget_session_keys()
        # Clear any existing knowledge graph visualisation
        with container_data_source.status(
            "Reading article and digesting the information...", expanded=True
        ):
            with st.container(height=UI_STATUS_CONTAINER_HEIGHT, border=0):
                start_time = time.time()
                if st.session_state.ui__radio_source_type == SOURCE_TYPE_WIKIPEDIA:
                    st.toast(
                        f":notebook: Looking for the {st.session_state.ui__txtinput_document_source} on Wikipedia in {format_wikipedia_languages(st.session_state.ui__selectbox_wikipedia_prefix)}."
                    )
                    reader = WikipediaReader()
                    if (
                        st.session_state.ui__selectbox_wikipedia_prefix
                        != st.session_state.const_default_wikipedia_language_prefix
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
                        f":newspaper: Fetched {len(documents)} Wikipedia article {'entry' if len(documents)==1 else 'entries'}."
                    )
                elif st.session_state.ui__radio_source_type == SOURCE_TYPE_ARXIV:
                    reader = ArxivReader()
                    documents = reader.load_data(
                        papers_dir="tmp-arxiv",
                        search_query=st.session_state.ui__txtinput_document_source,
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} entries from multiple arXiv papers."
                    )
                elif st.session_state.ui__radio_source_type == SOURCE_TYPE_PUBMED:
                    reader = PubmedReader()
                    documents = reader.load_data(
                        search_query=st.session_state.ui__txtinput_document_source
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} entries from multiple Pubmed articles."
                    )
                elif st.session_state.ui__radio_source_type == SOURCE_TYPE_WEBPAGE:
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
                elif st.session_state.ui__radio_source_type == SOURCE_TYPE_PDF:
                    documents = load_remote_pdf_data(
                        pdf_url=st.session_state.ui__txtinput_document_source
                    )
                    st.warning(
                        f":newspaper: Fetched {len(documents)} pages(s) from {st.session_state.ui__txtinput_document_source}."
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
                if (
                    not st.session_state.settings_graphdb__disable
                    and not st.session_state.settings_redis__disable
                ):
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
        copy_shadow_ui_widget_session_keys()
        # Clear any existing knowledge graph visualisation
        with container_data_source.status("Loading existing index...", expanded=True):
            with st.container(height=UI_STATUS_CONTAINER_HEIGHT, border=0):
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
        copy_shadow_ui_widget_session_keys()
        with container_data_source.status(
            "Creating graph visualisation...", expanded=True
        ):
            with st.container(height=UI_STATUS_CONTAINER_HEIGHT, border=0):
                start_time = time.time()
                st.warning(
                    f":snail: Building graph visualisation from {len(list(st.session_state.index.index_struct.table.keys()))} nodes. This MAY take time, if the graph is large!"
                )
                save_graph_visualisation(st.session_state.index)
                end_time = time.time()
                st.success(
                    f":white_check_mark: Done in {round(end_time-start_time)} second(s)! The graph visualisation can be seen in the **Knowledge Graph Visualisation** page in the sidebar on the left."
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
    key="shadow__ui__chk_source_rebuild_index",
    help="Check this box to clear the cache and rebuild index when the source document(s) have changed or you have changed the LLM.",
    on_change=copy_shadow_ui_widget_session_keys,
)

col2_source.button(
    "**Load** existing index, if using storage",
    disabled=st.session_state.shadow__ui__txtinput_document_source == ""
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
chatbot_heading, chatbot_status = col2_main.columns(2)
chatbot_heading.markdown("## Chat about the data")
status_placeholder = chatbot_status.empty()

if "chatbot" in st.session_state and st.session_state.chatbot is not None:
    container_chat = col2_main.container(height=UI_CHAT_CONTAINER_HEIGHT, border=1)
    for message in st.session_state.messages:  # Display the prior chat messages
        with container_chat.chat_message(message[CHAT_KEY_ROLE]):
            st.markdown(message[CHAT_KEY_CONTENT])

    if prompt := col2_main.chat_input("Type your question or message here"):
        st.session_state.messages.append(
            {
                CHAT_KEY_ROLE: CHAT_KEY_VALUE_USER,
                CHAT_KEY_CONTENT: prompt,
                CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            }
        )
        with container_chat.chat_message(CHAT_KEY_VALUE_USER):
            st.markdown(prompt)
        with container_chat.chat_message(CHAT_KEY_VALUE_ASSISTANT):
            status_placeholder.info(
                f":balloon: Using **{st.session_state.settings_llm__provider}**: {st.session_state.settings_llm__model}",
            )
            with st.empty():
                st.markdown("Figuring out how to respond...")
                response = st.write_stream(
                    stream_wrapper(st.session_state.chatbot.stream_chat(prompt))
                )
            status_placeholder.write("")
        # Add response to message history
        st.session_state.messages.append(
            {
                CHAT_KEY_ROLE: CHAT_KEY_VALUE_ASSISTANT,
                CHAT_KEY_CONTENT: response,
                CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            }
        )
else:
    col2_main.info(
        """:information_desk_person: Chatbot is not ready! Please build the index 
        from the source document(s) or reload an existing index to start chatting."""
    )


if st.session_state.langfuse_callback_handler is not None:
    st.session_state.langfuse_callback_handler.flush()
