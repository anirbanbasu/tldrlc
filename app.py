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

"""Main Solara app and routing."""

from typing import Any
import solara

import reacton.ipyvuetify as rv

import utils.state_manager as sm
import utils.constants as constants

import ui.settings as settings_uic
import ui.ingest as ingest_uic
import ui.chat as chat_uic


from pathlib import Path

CWD = Path(__file__).parent
extern_style = (CWD / "styles.css").read_text(encoding=constants.CHAR_ENCODING_UTF8)

page_step: solara.Reactive[int] = solara.reactive(1)


@solara.component
def CustomLayout(children: Any = []):
    sm.set_theme_colours()
    sm.initialise_default_settings()

    with solara.AppLayout(
        children=children,
        color=None,  # sm.corrective_background_colour.value,
        navigation=True,
        sidebar_open=False,
    ) as app_layout:
        with solara.AppBar():
            with rv.Btn(
                icon=True,
                tag="a",
                attributes={
                    "href": constants.PROJECT_GIT_REPO_URL,
                    "title": f"{constants.PROJECT_NAME} {constants.PROJECT_GIT_REPO_LABEL}",
                    "target": constants.HTTP_TARGET_BLANK,
                },
            ):
                rv.Icon(children=["mdi-github-circle"])
            solara.lab.ThemeToggle()
            with rv.Snackbar(
                timeout=0,
                multi_line=True,
                color=sm.status_message_colour.value,
                v_model=sm.status_message_show.value,
            ):
                solara.Markdown(f"{sm.status_message.value}")
    return app_layout


@solara.component
def Page():
    # Remove the "This website runs on Solara" message
    solara.Style(constants.UI_SOLARA_NOTICE_REMOVE)
    solara.Style(extern_style)

    step_labels = [1, 2, 3, 4]

    with solara.Sidebar():
        if page_step.value in step_labels[2:]:
            settings_uic.AllSettingsCategorical()

    with rv.Stepper(
        alt_labels=False,
        vertical=False,
        non_linear=False,
        v_model=page_step.value,
    ):
        with rv.StepperHeader():
            for step in step_labels:
                with rv.StepperStep(step=step, complete=page_step.value > step):
                    match step:
                        case 1:
                            solara.Markdown("Information")
                        case 2:
                            solara.Markdown("Language model (LLM)")
                        case 3:
                            solara.Markdown("Data")
                        case 4:
                            solara.Markdown("Chat")
                if step != step_labels[-1]:
                    rv.Divider()
        with rv.StepperItems():
            with rv.StepperContent(step=1):
                with rv.Card(elevation=0):
                    solara.Markdown(constants.MESSAGE_TLDRLC_WELCOME)
                    solara.Markdown(
                        f"**{constants.NOTICE_EU_AI_ACT__TITLE}**: {constants.NOTICE_EU_AI_ACT__MESSAGE}"
                    )
                with rv.CardActions():
                    solara.Button(
                        constants.BTN_NOTICE_EU_AI_ACT__MORE,
                        color="warning",
                        icon_name="mdi-github-circle",
                        attributes={
                            "href": constants.PROJECT_GIT_REPO_URL,
                            "title": f"{constants.PROJECT_NAME} {constants.PROJECT_GIT_REPO_LABEL}",
                            "target": constants.HTTP_TARGET_BLANK,
                        },
                    )
                    solara.Button(
                        constants.BTN_NOTICE_EU_AI_ACT__OK,
                        color="primary",
                        icon_name="mdi-thumb-up",
                        on_click=lambda: page_step.set(2),
                    )
            with rv.StepperContent(step=2):
                with rv.Card(elevation=0):
                    solara.Markdown(
                        """
                        ### Language model settings

                        _You can configure other settings of the language model along 
                        with indexing and storage from the settings menu, which is available
                        from the next step on the left sidebar_.
                        """,
                    )
                    settings_uic.LLMSettingsBasicComponent()
                    with rv.CardActions():
                        solara.Button(
                            constants.EMPTY_STRING,
                            icon_name="mdi-information",
                            on_click=lambda: page_step.set(1),
                        )
                        solara.Button(
                            "Get data",
                            icon_name="mdi-page-next",
                            color="primary",
                            on_click=lambda: page_step.set(3),
                        )
            with rv.StepperContent(step=3):
                with rv.Card(elevation=0):
                    solara.Markdown(
                        """
                        ### Data ingestion

                        You must ingest data from one of the following sources in order to chat about it.
                        """,
                    )
                    ingest_uic.IngestSelectiveComponent()
                    if (
                        sm.global_knowledge_graph_index.value is not None
                        and sm.global_semantic_search_index.value is not None
                    ) and not (
                        ingest_uic.ingest_webpage_data.pending
                        or ingest_uic.ingest_pdfurl_data.pending
                        or ingest_uic.ingest_wikipedia_data.pending
                        or ingest_uic.ingest_arxiv_data.pending
                        or ingest_uic.ingest_pubmed_data.pending
                    ):
                        solara.Markdown(
                            """If you are storing external graph and indices storage, 
                            you can use the following index IDs to reload the knowledge 
                            graph and semantic search indices that correspond to the last 
                            ingested data source."""
                        )
                        rv.Alert(
                            type="success",
                            outlined=True,
                            icon="mdi-graph-outline",
                            children=[
                                solara.Markdown(
                                    f"**Knowledge graph**: {sm.global_knowledge_graph_index.value.index_id}"
                                )
                            ],
                        )
                        rv.Alert(
                            type="success",
                            outlined=True,
                            icon="mdi-vector-combine",
                            children=[
                                solara.Markdown(
                                    f"**Semantic search**: {sm.global_semantic_search_index.value.index_id}"
                                )
                            ],
                        )
                    with rv.CardActions():
                        solara.Button(
                            "LLM",
                            icon_name="mdi-cogs",
                            disabled=(
                                ingest_uic.ingest_webpage_data.pending
                                or ingest_uic.ingest_pdfurl_data.pending
                                or ingest_uic.ingest_wikipedia_data.pending
                                or ingest_uic.ingest_arxiv_data.pending
                                or ingest_uic.ingest_pubmed_data.pending
                            ),
                            on_click=lambda: page_step.set(2),
                        )
                        solara.Button(
                            "Let's chat!",
                            icon_name="mdi-chat-processing",
                            color="primary",
                            disabled=(
                                ingest_uic.ingest_webpage_data.pending
                                or ingest_uic.ingest_pdfurl_data.pending
                                or ingest_uic.ingest_wikipedia_data.pending
                                or ingest_uic.ingest_arxiv_data.pending
                                or ingest_uic.ingest_pubmed_data.pending
                                or ingest_uic.last_ingested_data_source.value
                                == constants.EMPTY_STRING
                            ),
                            on_click=lambda: page_step.set(4),
                        )
            with rv.StepperContent(step=4):
                with rv.Card(elevation=0):
                    with rv.CardActions():
                        solara.Button(
                            "Go back to get some other data",
                            color="primary",
                            outlined=True,
                            icon_name="mdi-page-previous",
                            on_click=lambda: page_step.set(3),
                        )
                chat_uic.ChatInterface()


routes = [
    solara.Route(path="/", component=Page, label="TLDRLC", layout=CustomLayout),
]
