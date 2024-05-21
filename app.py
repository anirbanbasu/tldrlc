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
from solara.alias import rv

from pages import chatbot, ingest, settings
from utils import global_state


@solara.component
def CustomLayout(children: Any = []):
    global_state.set_theme_colours()
    global_state.initialise_default_settings()

    with solara.AppLayout(
        children=children,
        color=global_state.corrective_background_colour.value,
        navigation=True,
        sidebar_open=False,
    ) as app_layout:
        with solara.AppBar():
            with solara.v.Btn(
                icon=True,
                tag="a",
                attributes={
                    "href": "https://github.com/anirbanbasu/tldrlc",
                    "title": "TLDRLC GitHub repository",
                    "target": "_blank",
                },
            ):
                solara.v.Icon(children=["mdi-github-circle"])
            solara.lab.ThemeToggle()
            with rv.Snackbar(
                bottom=True,
                left=True,
                timeout=0,
                multi_line=True,
                color=global_state.status_message_colour.value,
                v_model=global_state.status_message_show.value,
            ):
                solara.Markdown(f"{global_state.status_message.value}")
    return app_layout


routes = [
    solara.Route(
        path="/", component=chatbot.Page, label="Chatbot", layout=CustomLayout
    ),
    solara.Route(path="ingest", component=ingest.Page, label="Ingest data"),
    solara.Route(path="settings", component=settings.Page, label="Settings"),
]
