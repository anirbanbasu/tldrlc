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

"""Various custom retrievers."""

# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
)

from typing import List

import utils.constants as constants


class VectorKnowledgeGraphRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and knowledge graph search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        knowledge_graph_retriever: KGTableRetriever,
        mode: str = constants.BOOLEAN_OR,
    ) -> None:
        """Init params."""

        if vector_retriever is None:
            raise ValueError("A valid vector index retriever must be specified.")
        if knowledge_graph_retriever is None:
            raise ValueError("A valid knowledge graph retriever must be specified.")
        if mode not in (constants.BOOLEAN_AND, constants.BOOLEAN_OR):
            raise ValueError(
                f"Invalid retriever mode {mode}. It must be either {constants.BOOLEAN_AND} or {constants.BOOLEAN_OR}."
            )
        self._vector_retriever = vector_retriever
        self._keyword_retriever = knowledge_graph_retriever
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        knowledge_graph_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        knowledge_graph_ids = {n.node.node_id for n in knowledge_graph_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in knowledge_graph_nodes})

        if self._mode == constants.BOOLEAN_AND:
            retrieve_ids = vector_ids.intersection(knowledge_graph_ids)
        elif self._mode == constants.BOOLEAN_OR:
            retrieve_ids = vector_ids.union(knowledge_graph_ids)
        else:
            raise ValueError(
                f"Set operation is not defined for invalid retriever mode {self._mode}, which must be either {constants.BOOLEAN_AND} or {constants.BOOLEAN_OR}."
            )

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
