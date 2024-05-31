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
import uuid

import solara
import solara.lab
from solara.lab import task  # , Task, use_task
from solara.alias import rv

import utils.constants as constants
import utils.state_manager as sm
from utils.state_manager import show_status_message
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from langfuse.decorators import observe, langfuse_context

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

# Reactive variables
langfuse_last_trace_id: solara.Reactive[str] = solara.Reactive(constants.EMPTY_STRING)
langfuse_last_observation_id: solara.Reactive[str] = solara.Reactive(
    constants.EMPTY_STRING
)
chat_session_id: solara.Reactive[str] = solara.Reactive(constants.EMPTY_STRING)
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


def clear_chat_history():
    """Clear the chat history."""
    sm.global_chat_messages.value.clear()


def no_chat_engine_message():
    sm.global_chat_messages.value = [
        {
            constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
            constants.CHAT_KEY_CONTENT: "_The chat engine is **not available**. Please check the settings and then ingest some data from the Ingest page to initialise the chat engine._",
            constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            constants.CHAT_KEY_LLM_PROVIDER: "TLDRLC",
            constants.CHAT_KEY_LLM_MODEL_NAME: "System",
        },
    ]


@task(prefer_threaded=True)
def submit_feedback(callback_arg=None):
    try:
        if (
            langfuse_last_trace_id.value is None
            or langfuse_last_observation_id.value is None
        ):
            raise ValueError("No associated Langfuse trace and observation was found.")
        sm.global_client_langfuse_lowlevel.value.score(
            name="Human feedback",
            value=ai_response_feedback_score.value,
            comment=ai_response_feedback_comment.value,
            trace_id=langfuse_last_trace_id.value,
            observation_id=langfuse_last_observation_id.value,
        )
        sm.global_client_langfuse_lowlevel.value.flush()
        logger.info(
            f"Feedback submitted: Score: {ai_response_feedback_score.value}, Comment: {ai_response_feedback_comment.value}."
        )
        show_status_message(
            "Feedback submitted, thank you!",
            colour="success",
        )
        # Reset the feedback form
        ai_response_feedback_score.value = 0.0
        ai_response_feedback_comment.value = constants.EMPTY_STRING
        langfuse_last_trace_id.value = None
        langfuse_last_observation_id.value = None
    except Exception as e:
        logger.error(f"Error submitting feedback. {e}")
        show_status_message(f"Error submitting feedback. {e}", colour="error")


def add_chunk_to_ai_message(chunk: str):
    sm.global_chat_messages.value = [
        *sm.global_chat_messages.value[:-1],
        {
            constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
            constants.CHAT_KEY_CONTENT: sm.global_chat_messages.value[-1][
                constants.CHAT_KEY_CONTENT
            ]
            + chunk,
            constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
            constants.CHAT_KEY_LLM_PROVIDER: sm.global_settings__language_model_provider.value[
                :
            ],
            constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[:],
        },
    ]


@task(prefer_threaded=True)
@observe()
def ask_tldrlc(callback_arg=None) -> bool:
    result = False
    langfuse_last_trace_id.value = None
    if sm.global_chat_engine.value is not None:
        # Generate a new session ID if the chat history is empty or it contains just the system chat-engine-not-ready message.
        if len(sm.global_chat_messages.value) <= 1:
            chat_session_id.value = str(uuid.uuid4())
        message = user_chat_input.value
        if message is not None and len(message) > 0:
            if len(sm.global_chat_messages.value) == 1:
                # Remove the only not-initialised status message from the chatbot
                sm.global_chat_messages.value = [
                    {
                        constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_USER,
                        constants.CHAT_KEY_CONTENT: message,
                        constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                    },
                ]
            else:
                sm.global_chat_messages.value = [
                    *sm.global_chat_messages.value,
                    {
                        constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_USER,
                        constants.CHAT_KEY_CONTENT: message,
                        constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                    },
                ]
            last_response_ai.value = None
            user_chat_input.value = constants.EMPTY_STRING
            result = call_chat_engine()
            langfuse_context.update_current_trace(
                tags=sm.global_settings_langfuse_tags.value,
                session_id=chat_session_id.value,
            )
            langfuse_last_trace_id.value = langfuse_context.get_current_trace_id()
            result = result and True
    else:
        no_chat_engine_message()
    return result


@observe(as_type="generation")
def call_chat_engine() -> bool:
    langfuse_last_observation_id.value = None
    has_unanswered_messages = (
        len(sm.global_chat_messages.value) > 0
        and sm.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
        == constants.CHAT_KEY_VALUE_USER
    )
    try:
        if not has_unanswered_messages:
            return
        user_query = sm.global_chat_messages.value[-1][constants.CHAT_KEY_CONTENT]
        last_response_ai.value = sm.global_chat_engine.value.stream_chat(
            message=user_query,
        )
        sm.global_chat_messages.value = [
            *sm.global_chat_messages.value,
            {
                constants.CHAT_KEY_ROLE: constants.CHAT_KEY_VALUE_ASSISTANT,
                constants.CHAT_KEY_CONTENT: constants.EMPTY_STRING,
                constants.CHAT_KEY_TIMESTAMP: f"{datetime.datetime.now()}",
                constants.CHAT_KEY_LLM_PROVIDER: sm.global_settings__language_model_provider.value[
                    :
                ],
                constants.CHAT_KEY_LLM_MODEL_NAME: Settings.llm.metadata.model_name[:],
            },
        ]
        # for node in last_response_ai.value.source_nodes:
        #     logger.warning(f"Source node: {node.metadata}")
        for chunk in last_response_ai.value.response_gen:
            add_chunk_to_ai_message(chunk)

        langfuse_context.update_current_observation(
            input=user_query,
            output=last_response_ai.value.response,
            model=Settings.llm.metadata.model_name,
            metadata=Settings.llm.metadata,
            session_id=chat_session_id.value,
            tags=sm.global_settings_langfuse_tags.value,
        )
        langfuse_last_observation_id.value = (
            langfuse_context.get_current_observation_id()
        )
        return True
    except Exception as e:
        logger.error(f"Error with chat engine. {e}")
        show_status_message(f"Error with chat engine. {e}", colour="error")
        langfuse_last_observation_id.value = None
        return False


@solara.component
def ChatInterface():
    with solara.Column(
        style={
            "width": "95vw",
            "margin": "auto",
            "padding-left": "1em",
            "padding-right": "2em",
            "bottom": "1%",
            "position": "fixed",
        }
    ):
        with solara.Column(
            style={
                "width": "100%",
                "margin": "auto",
                "max-height": "65vh",
                "overflow-y": "auto",
                "overflow-x": "hidden",
            }
        ):
            with solara.lab.ChatBox():
                for item in sm.global_chat_messages.value:
                    with solara.lab.ChatMessage(
                        user=item[constants.CHAT_KEY_ROLE]
                        == constants.CHAT_KEY_VALUE_USER,
                        avatar=(
                            "mdi-account"
                            if item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_USER
                            else "mdi-robot"
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
                        (
                            solara.Markdown(
                                md_text=":thinking: _Thinking of a response..._"
                            )
                            if item[constants.CHAT_KEY_CONTENT]
                            == constants.EMPTY_STRING
                            else solara.Markdown(
                                md_text=f"{item[constants.CHAT_KEY_CONTENT]}"
                            )
                        )
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
                            # Show feedback form only if Langfuse is enabled and the last message of a conversation was from the assistant.
                            item[constants.CHAT_KEY_ROLE]
                            == constants.CHAT_KEY_VALUE_ASSISTANT
                            and item == sm.global_chat_messages.value[-1]
                            and not ask_tldrlc.pending
                            and not submit_feedback.pending
                            and len(sm.global_chat_messages.value) > 1
                            and sm.global_settings_langfuse_enabled.value
                            and (
                                langfuse_last_trace_id.value is not None
                                and langfuse_last_observation_id.value is not None
                            )
                        ):
                            with rv.ExpansionPanels():
                                with rv.ExpansionPanel(
                                    disabled=submit_feedback.pending
                                ):
                                    with rv.ExpansionPanelHeader():
                                        solara.Markdown("🧐 _How did I do?_")
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
            if (
                len(sm.global_chat_messages.value) > 0
                and sm.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
                == constants.CHAT_KEY_VALUE_USER
            ):
                solara.Markdown(":construction: _Working on it..._")
            # if exported_chat_json.value:
            #     solara.Markdown(
            #         f"""
            #         ```json
            #         {exported_chat_json.value}
            #         ```
            #         """
            #     )
        with solara.Row():
            solara.InputText(
                label="Type your message here...",
                style={"width": "100%"},
                value=user_chat_input,
                update_events=["keyup.enter"],
                on_value=ask_tldrlc,
                disabled=(
                    ask_tldrlc.pending
                    or (
                        len(sm.global_chat_messages.value) > 0
                        and sm.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
                        == constants.CHAT_KEY_VALUE_USER
                    )
                ),
            )
            solara.Button(
                label="Ask",
                icon_name="mdi-send",
                on_click=ask_tldrlc,
                color="success",
                disabled=(
                    ask_tldrlc.pending
                    or (
                        len(sm.global_chat_messages.value) > 0
                        and sm.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
                        == constants.CHAT_KEY_VALUE_USER
                    )
                ),
            )
            # solara.Button(
            #     label="Clear chat",
            #     on_click=clear_chat_history,
            #     color="error",
            #     disabled=(
            #         (ask_tldrlc.pending or len(sm.global_chat_messages.value) == 0)
            #         or (
            #             len(sm.global_chat_messages.value) > 0
            #             and sm.global_chat_messages.value[-1][constants.CHAT_KEY_ROLE]
            #             == constants.CHAT_KEY_VALUE_USER
            #         )
            #     ),
            # )
