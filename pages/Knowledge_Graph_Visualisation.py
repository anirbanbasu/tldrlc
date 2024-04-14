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

"""Module for the visualisation of the knowledge graph generated from the index in the TLDRLC application."""

import streamlit as st
from utils.streamlit.settings_manager import load_settings
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

st.title("TLDRLC Knowledge Graph Visualisation")

# Session state metadata cannot be used from the other pages if they are connected to widgest.
# See: https://docs.streamlit.io/library/advanced-features/widget-behavior#save-widget-values-in-session-state-to-preserve-them-between-pages

with st.sidebar:
    write_eu_ai_act_transparency_notice()

st.markdown(
    """
    The graph displayed below is generated from a knowledge graph index. 
    This graph may not be an exhaustive representation of the actual knowledge 
    graph used in the TLDRLC application because the number of nodes 
    displayed in the visualisation has been limited to a maximum value for efficiency. 
    The graph is interactive and can be zoomed in/out and panned. 
    Edge metadata can be viewed by selecting an edges.

    :bulb: To ensure that you are seeing the visualisation of the graph 
    of the presently used index, press the `Visualise knowledge graph` button 
    in the TLDRLC Chatbot application.
    """
)

if "data__kg_html" in st.session_state.keys():
    st.components.v1.html(
        st.session_state.data__kg_html,
        height=st.session_state.settings_saved_kgvis__height + 25,
    )
else:
    st.error(
        ":x: No knowledge graph data found. Please generate the knowledge graph first."
    )
