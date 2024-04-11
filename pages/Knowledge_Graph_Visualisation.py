# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=broad-exception-caught

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

# Setup page metadata
st.set_page_config(
    page_title="TLDRLC Knowledge Graph Visualisation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔗",
    menu_items={
        "About": """
        This page is an interactive visualisation of a knowledge graph generated from the index in the
        Too Long, Didn't Read, Let's Chat (TLDRLC) application. Check out the project 
        on [Github](https://github.com/anirbanbasu/tldrlc).
        """,
        "Report a Bug": "https://github.com/anirbanbasu/tldrlc/issues",
    },
)

st.title("TLDRLC Knowledge Graph Visualisation")

# Session state metadata cannot be used from the other pages if they are connected to widgest.
# See: https://docs.streamlit.io/library/advanced-features/widget-behavior#save-widget-values-in-session-state-to-preserve-them-between-pages


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

    **EU AI Act [Article 52](https://artificialintelligenceact.eu/article/52/) Transparency notice**: :red[By using this app, you 
    are interacting with an artificial intelligence (AI) system. You are advised not to take this AI-generated knowledge graph as a fact.
    The AI system is not a substitute for professional advice. If you are unsure about any information, please consult a professional in the field.]
    """
)

try:
    with open(
        st.session_state.settings_saved_kgvis__filename, "r", encoding="utf-8"
    ) as html_file:
        source_code = html_file.read()
        html_file.close()
    st.components.v1.html(
        source_code, height=st.session_state.settings_saved_kgvis__height + 25
    )
except Exception as kgvis_e:
    st.error(f":x: Error: {type(kgvis_e).__name__}: {kgvis_e}")
