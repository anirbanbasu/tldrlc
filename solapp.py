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

import solara

from pages import chatbot, ingest, settings

routes = [
    solara.Route(path="/", component=chatbot.Page, label="Chatbot"),
    solara.Route(path="ingest", component=ingest.Page, label="Ingest data"),
    solara.Route(path="settings", component=settings.Page, label="Settings"),
]
