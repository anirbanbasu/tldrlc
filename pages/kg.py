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

"""Knowledge graph visualisation page."""

import os
import logging

import solara
import solara.lab
from solara.lab import task

import utils.constants as constants
import utils.global_state as global_state

from pyvis.network import Network
import networkx as nx

logger = logging.getLogger(__name__)
# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.basicConfig(
    # See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
    format=constants.LOG_FORMAT,
    level=int(os.getenv(constants.ENV_KEY_LOG_LEVEL)),
    encoding=constants.CHAR_ENCODING_UTF8,
)

graph_visualisation_html: solara.Reactive[str] = solara.Reactive("")


@task
def generate_graph_visualisation() -> str:
    """Save the graph visualisation to HTML."""
    # Try to make a graph from the index at all depths, but limit the number of nodes
    graph = nx.Graph()
    already_added_nodes = set()
    nodes = list(
        global_state.global_knowledge_graph_index.value.index_struct.table.keys()
    )[slice(global_state.global_settings__kg_vis_max_nodes.value)]

    for node in nodes:
        triplets = global_state.global_knowledge_graph_index.value.storage_context.graph_store.get(
            node
        )
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
    match global_state.global_settings__kg_vis_layout.value:
        case constants.GRAPH_VIS_LAYOUT_CIRCULAR:
            positions = nx.circular_layout(graph, scale=_scale_factor)
        case constants.GRAPH_VIS_LAYOUT_PLANAR:
            positions = nx.planar_layout(graph, scale=_scale_factor)
        case constants.GRAPH_VIS_LAYOUT_SHELL:
            positions = nx.shell_layout(graph, scale=_scale_factor)
        case constants.GRAPH_VIS_LAYOUT_SPECTRAL:
            positions = nx.spectral_layout(graph, scale=_scale_factor)
        case constants.GRAPH_VIS_LAYOUT_SPRING:
            positions = nx.spring_layout(graph, scale=_scale_factor)
        case _:
            positions = nx.spring_layout(graph, scale=_scale_factor)

    network = Network(
        notebook=False,
        height=f"{global_state.global_settings__kg_vis_height.value}px",
        width="100%",
        cdn_resources="in_line",
        neighborhood_highlight=True,
        directed=True,
    )
    network.toggle_physics(global_state.global_settings__kg_vis_physics_enabled.value)
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
    graph_visualisation_html.value = network.generate_html()


@solara.component
def Page():
    # Remove the "This website runs on Solara" message
    solara.Style(constants.UI_SOLARA_NOTICE_REMOVE)

    global_state.initialise_default_settings()

    with solara.AppBarTitle():
        solara.Markdown("# Knowledge Graph Visualisation", style={"color": "#FFFFFF"})

    with solara.AppBar():
        solara.lab.ThemeToggle()

    with solara.Column(
        style={
            "width": "100%",
            "padding-left": "1em",
            "padding-right": "2em",
            "bottom": (
                "1%"
                if global_state.global_knowledge_graph_index.value is not None
                else "auto"
            ),
            "position": "fixed",
        }
    ):
        with solara.Column(
            style={
                "width": "100%",
                "margin-top": "auto",
                "margin-bottom": "auto",
                "max-height": "80vh",
                "overflow-y": "auto",
                "overflow-x": "auto",
            }
        ):
            if global_state.global_knowledge_graph_index.value is not None:
                task_generate_graph = solara.lab.use_task(generate_graph_visualisation)
                solara.ProgressLinear(task_generate_graph.pending)
                if task_generate_graph.finished:
                    solara.Info(
                        label="Knowledge graph visualisation has not been correctly implemented yet."
                    )
                    solara.HTML(
                        unsafe_innerHTML=f"<iframe csp='unsafe-inline' src='about:blank' style='margin-bottom: auto; margin-top: auto; width: 100%; height: 100%; overflow: auto; border: 0' srcdoc='{graph_visualisation_html.value}' width='100%' height='100%'></iframe>"
                    )

            else:
                solara.Warning(
                    label="No knowledge graph data is available. Ingest some data to initialise a knowledge graph."
                )
        solara.Markdown(
            """
            :warning: **EU AI Act [Article 52](https://artificialintelligenceact.eu/article/52/) Transparency notice**:
            By using this app, you are interacting with an artificial intelligence (AI) system. 
            <u>You are advised not to take any of its responses as facts</u>. The AI system is not a 
            substitute for professional advice. If you are unsure about any information, please 
            consult a professional in the field.
            """,
            style={
                "margin-left": "auto",
                "margin-right": "auto",
                "width": "100%",
                "border-top": "1px solid black",
                "border-radius": "0px",
            },
        )
