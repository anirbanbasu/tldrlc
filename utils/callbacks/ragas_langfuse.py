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

"""Custom Langfuse callback handler for use with Ragas."""

from typing import Any, Dict
from llama_index.core.callbacks.schema import CBEventType
from langfuse.llama_index import LlamaIndexCallbackHandler


class RagasLangfuseCallbackHandler(LlamaIndexCallbackHandler):
    """Custom Langfuse callback handler to use Ragas evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return super().on_event_start(
            event_type=event_type,
            payload=payload,
            event_id=event_id,
            parent_id=parent_id,
            **kwargs,
        )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().on_event_end(
            event_type=event_type, payload=payload, event_id=event_id, **kwargs
        )
