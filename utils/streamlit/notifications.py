# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=duplicate-code
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

"""Module for managing various notifications of the application."""

import streamlit as st


def write_eu_ai_act_transparency_notice():
    """Writes the notice for the European Union's AI Act transparency requirements."""
    st.markdown(
        """
        ## :warning: EU AI Act [Article 52](https://artificialintelligenceact.eu/article/52/) Transparency notice
        
        By using this app, you  are interacting with an artificial intelligence (AI) system.
        You are advised not to  take any of its responses as facts. The AI system is not a substitute 
        for professional advice. If you are unsure about any information, please consult a professional in the field.
        """
    )
