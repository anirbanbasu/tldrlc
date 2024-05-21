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
from solara.alias import rv

import utils.constants as constants
import utils.global_state as global_state
from utils.global_state import show_status_message
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

logger = logging.getLogger(__name__)
# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.basicConfig(
    # See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
    format=constants.LOG_FORMAT,
    level=int(
        os.getenv(constants.ENV_KEY_LOG_LEVEL, constants.DEFAULT_SETTING_LOG_LEVEL)
    ),
    encoding=constants.CHAR_ENCODING_UTF8,
)

user_chat_input: solara.Reactive[str] = solara.Reactive(constants.EMPTY_STRING)
exported_chat_json: solara.Reactive[str] = solara.Reactive(constants.EMPTY_STRING)
last_response_ai: solara.Reactive[StreamingAgentChatResponse] = solara.Reactive(None)
ai_response_feedback_score: solara.Reactive[float] = solara.Reactive(0.0)
ai_response_feedback_comment: solara.Reactive[str] = solara.Reactive(
    constants.EMPTY_STRING
)


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

    has_unanswered_messages = (
        len(global_state.global_chat_messages.value) > 0
        and global_state.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
        == constants.CHAT_KEY_VALUE_USER
    )

    def clear_chat_history():
        """Clear the chat history."""
        global_state.global_chat_messages.value = []

    def no_chat_engine_message():
        global_state.global_chat_messages.value = [
            {
                constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
                constants.CHAT_KEY_CONTENT: "_The chat engine is **not available**. Please check the settings and then ingest some data from the Ingest page to initialise the chat engine._",
                constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                constants.CHAT_KEY_LLM_PROVIDER: "TLDRLC",
                constants.CHAT_KEY_LLM_MODEL_NAME: "System",
            },
        ]

    def submit_feedback(callback_arg=None):
        show_status_message(
            "Feedback submission has not been enabled yet. Please check back later.",
            colour="warning",
        )

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

    def ask_tldrlc(callback_arg=None):
        message = user_chat_input.value
        if message is not None and len(message) > 0:
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
            last_response_ai.value = None
            user_chat_input.value = constants.EMPTY_STRING

    def call_chat_engine():
        try:
            if not has_unanswered_messages:
                return
            if global_state.global_chat_engine.value is None:
                no_chat_engine_message()
                return
            last_response_ai.value = global_state.global_chat_engine.value.stream_chat(
                message=global_state.global_chat_messages.value[-1][
                    constants.CHAT_KEY_CONTENT
                ],
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
                    constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[
                        :
                    ],
                },
            ]
            # for node in last_response_ai.value.source_nodes:
            #     logger.warning(f"Source node: {node.metadata}")
            for chunk in last_response_ai.value.response_gen:
                add_chunk_to_ai_message(chunk)
        except Exception as e:
            logger.error(f"Error with chat engine. {e}")
            show_status_message(f"Error with chat engine. {e}", colour="error")

    task_get_chat_response = solara.lab.use_task(
        call_chat_engine, dependencies=[has_unanswered_messages]
    )

    with solara.AppBarTitle():
        solara.Text(
            "TL;DR Let's Chat",
        )

    # with rv.Snackbar(
    #     top=True,
    #     right=True,
    #     timeout=0,
    #     multi_line=True,
    #     color=global_state.status_message_colour.value,
    #     v_model=global_state.status_message_show.value,
    # ):
    #     solara.Markdown(f"{global_state.status_message.value}")

    with solara.lab.ConfirmationDialog(
        open=global_state.show_eu_ai_act_notice,
        ok=constants.NOTICE_EU_AI_ACT__OK,
        cancel=solara.Button(
            constants.NOTICE_EU_AI_ACT__CANCEL, disabled=True, style={"display": "none"}
        ),
        title=constants.NOTICE_EU_AI_ACT__TITLE,
        on_ok=lambda: global_state.show_eu_ai_act_notice.set(False),
        persistent=True,
    ):
        solara.Markdown(
            constants.NOTICE_EU_AI_ACT__MESSAGE,
        )

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
                            (
                                solara.lab.theme.themes.dark.accent
                                if solara.lab.use_dark_effective()
                                else (solara.lab.theme.themes.light.accent)
                            )
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            else (
                                solara.lab.theme.themes.dark.secondary
                                if solara.lab.use_dark_effective()
                                else (solara.lab.theme.themes.light.secondary)
                            )
                        ),
                        avatar_background_color=(
                            (
                                solara.lab.theme.themes.dark.accent
                                if solara.lab.use_dark_effective()
                                else (solara.lab.theme.themes.light.accent)
                            )
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            else (
                                solara.lab.theme.themes.dark.secondary
                                if solara.lab.use_dark_effective()
                                else (solara.lab.theme.themes.light.secondary)
                            )
                        ),
                        border_radius="8px",
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
                        if (
                            item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            and item == global_state.global_chat_messages.value[-1]
                            and task_get_chat_response.finished
                            and len(global_state.global_chat_messages.value) > 1
                            and global_state.global_settings_langfuse_enabled.value
                        ):
                            with rv.ExpansionPanels():
                                with rv.ExpansionPanel():
                                    with rv.ExpansionPanelHeader():
                                        solara.Markdown(
                                            ":monocle_face: _How did I do?_"
                                        )
                                    with rv.ExpansionPanelContent():
                                        solara.SliderFloat(
                                            label="Score",
                                            tick_labels=False,
                                            step=0.1,
                                            min=-1.0,
                                            max=1.0,
                                            value=ai_response_feedback_score,
                                        )
                                        solara.InputText(
                                            label="Comments",
                                            style={"width": "100%"},
                                            value=ai_response_feedback_comment,
                                        )
                                        solara.Button(
                                            label="Submit",
                                            color="success",
                                            outlined=True,
                                            on_click=submit_feedback,
                                        )
            if task_get_chat_response.pending:
                solara.Markdown(":thinking: _Thinking of a response..._")
            if exported_chat_json.value:
                solara.Markdown(
                    f"""
                    ```json
                    {exported_chat_json.value}
                    ```        
                    """
                )
        with solara.Row():
            solara.InputText(
                label="Type your message here...",
                style={"width": "100%"},
                value=user_chat_input,
                on_value=ask_tldrlc,
                disabled=task_get_chat_response.pending,
            )
            solara.Button(
                label="Ask",
                icon_name="mdi-send",
                on_click=ask_tldrlc,
                color="success",
                disabled=task_get_chat_response.pending,
                # outlined=True,
            )
            solara.Button(
                label="Clear chat",
                on_click=clear_chat_history,
                color="error",
                disabled=(
                    task_get_chat_response.pending
                    or len(global_state.global_chat_messages.value) == 0
                ),
                style={"display": "none"},
            )
