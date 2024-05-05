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

"""Chatbot page."""

import datetime
import os
import logging

import solara
import solara.lab

import utils.constants as constants
import utils.global_state as global_state
from llama_index.core import Settings

logger = logging.getLogger(__name__)
# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.basicConfig(
    # See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
    format=constants.LOG_FORMAT,
    level=int(os.getenv(constants.ENV_KEY_LOG_LEVEL)),
    encoding=constants.CHAR_ENCODING_UTF8,
)


def no_chat_engine_message():
    global_state.global_chat_messages.value = [
        {
            constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
            constants.CHAT_KEY_CONTENT: "_**The chat engine is not available. Please check the settings and then ingest some data to initialise the chat engine.**_",
            constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            constants.CHAT_KEY_LLM_PROVIDER: global_state.global_settings__language_model_provider.value[
                :
            ],
            constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[:],
        },
    ]


def add_chunk_to_ai_message(chunk: str):
    global_state.global_chat_messages.value = [
        *global_state.global_chat_messages.value[:-1],
        {
            constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
            constants.CHAT_KEY_CONTENT: global_state.global_chat_messages.value[-1][
                constants.CHAT_KEY_CONTENT
            ]
            + chunk,
            constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            constants.CHAT_KEY_LLM_PROVIDER: global_state.global_settings__language_model_provider.value[
                :
            ],
            constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[:],
        },
    ]


def stream_wrapper(streaming_response):
    """Wrapper for the streaming response from the chat engine."""
    for token in streaming_response.response_gen:
        # Filter out symbols that break formatting
        if token.strip() == "$":
            # Escape the $ sign, otherwise LaTeX formatting will be triggered!
            yield token.replace("$", "${$}")
        else:
            yield token


@solara.component
def Page():
    # Remove the "This website runs on Solara" message
    solara.Style(constants.UI_SOLARA_NOTICE_REMOVE)

    global_state.initialise_default_settings()

    has_unanswered_messages = (
        len(global_state.global_chat_messages.value) > 0
        and global_state.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
        == constants.CHAT_KEY_VALUE_USER
    )

    def ask_tldrlc(message):
        if len(global_state.global_chat_messages.value) == 1:
            # Remove the only not-initialised status message from the chatbot
            global_state.global_chat_messages.value = [
                {
                    constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_USER,
                    constants.CHAT_KEY_CONTENT: message,
                    constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                },
            ]
        else:
            global_state.global_chat_messages.value = [
                *global_state.global_chat_messages.value,
                {
                    constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_USER,
                    constants.CHAT_KEY_CONTENT: message,
                    constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                },
            ]

    def call_chat_engine():
        if not has_unanswered_messages:
            return
        if global_state.global_chat_engine.value is None:
            no_chat_engine_message()
            return
        response = global_state.global_chat_engine.value.stream_chat(
            global_state.global_chat_messages.value[-1][constants.CHAT_KEY_CONTENT]
        )
        global_state.global_chat_messages.value = [
            *global_state.global_chat_messages.value,
            {
                constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
                constants.CHAT_KEY_CONTENT: constants.EMPTY_STRING,
                constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                constants.CHAT_KEY_LLM_PROVIDER: global_state.global_settings__language_model_provider.value[
                    :
                ],
                constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[:],
            },
        ]
        for chunk in response.response_gen:
            add_chunk_to_ai_message(chunk)

    task_get_chat_response = solara.lab.use_task(
        call_chat_engine, dependencies=[has_unanswered_messages]
    )

    with solara.AppBarTitle():
        solara.Markdown(
            "# Too Long, Didn't Read, Let's Chat", style={"color": "#FFFFFF"}
        )

    with solara.AppBar():
        solara.lab.ThemeToggle()

    with solara.Column(
        style={
            "width": "100%",
            "padding-left": "1em",
            "padding-right": "2em",
            "bottom": "1%",
            "position": "fixed",
        }
    ):
        with solara.Column(
            style={
                "width": "100%",
                "max-height": "80vh",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            }
        ):
            with solara.lab.ChatBox():
                for item in global_state.global_chat_messages.value:
                    with solara.lab.ChatMessage(
                        user=item[constants.CHAT_KEY_ROLE]
                        == constants.CHAT_KEY_VALUE_USER,
                        avatar=(
                            "mdi-account"
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_USER
                            else "mdi-robot"
                        ),
                        name=(
                            "TLDRLC"
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            else "Human"
                        ),
                        color=(
                            "#85C1E9"
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            else "#F7DC6F"
                        ),
                        avatar_background_color=(
                            "#D6EAF8"
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            else "#FCF3CF"
                        ),
                        border_radius="16px",
                        notch=True,
                    ):
                        solara.Markdown(md_text=f"{item[constants.CHAT_KEY_CONTENT]}")
                        solara.Markdown(
                            md_text=(
                                f"<div style='text-align: right; font-size: 0.75em; padding-top: 1em'>{item[constants.CHAT_KEY_LLM_PROVIDER]}:{item[constants.CHAT_KEY_LLM_MODEL_NAME]}@{item[constants.CHAT_KEY_TIMESTAMP]}</div>"
                                if item[constants.CHAT_KEY_ROLE]
                                == constants.CHAT_KEY_VALUE_ASSISTANT
                                else f"<div style='text-align: right; font-size: 0.75em; padding-top: 1em'>{item[constants.CHAT_KEY_TIMESTAMP]}</div>"
                            ),
                            style={
                                "font-size": "0.75em",
                                "text-align": "right",
                                "padding-top": "1em",
                            },
                        )
            if task_get_chat_response.pending:
                solara.Markdown(":thinking: _Thinking of a response..._")
        with solara.Row(
            justify="space-between",
            style={"border-top": "1px solid black", "border-radius": "0px"},
        ):
            solara.Markdown(
                """
                :warning: **EU AI Act [Article 52](https://artificialintelligenceact.eu/article/52/) Transparency notice**:
                By using this app, you are interacting with an artificial intelligence (AI) system. 
                <u>You are advised not to take any of its responses as facts</u>. The AI system is not a 
                substitute for professional advice. If you are unsure about any information, please 
                consult a professional in the field.
                """,
            )
            solara.lab.ChatInput(
                send_callback=ask_tldrlc,
                disabled=task_get_chat_response.pending,
                style={"width": "100%"},
            )
