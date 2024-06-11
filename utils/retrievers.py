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
from llama_index.core.retrievers import BaseRetriever

from typing import List

import utils.constants as constants


class VectorKnowledgeGraphRetriever(BaseRetriever):
    """Custom retriever that retrieves from a semantic search (vector) index and a knowledge graph index."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        knowledge_graph_retriever: BaseRetriever,
        mode: str = constants.BOOLEAN_OR,
    ) -> None:
        """Initialisation parameters."""

        if vector_retriever is None:
            raise ValueError(
                "A valid semantic search (vector) index retriever must be specified."
            )
        if knowledge_graph_retriever is None:
            raise ValueError("A valid knowledge graph retriever must be specified.")
        if mode not in (constants.BOOLEAN_AND, constants.BOOLEAN_OR):
            raise ValueError(
                f"""Invalid retriever logical combination mode {mode}. 
                It must be either {constants.BOOLEAN_AND} (intersection) or {constants.BOOLEAN_OR} (union)."""
            )
        self._vector_retriever = vector_retriever
        self._keyword_retriever = knowledge_graph_retriever
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # Retrieve nodes from both indices
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        knowledge_graph_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        knowledge_graph_ids = {n.node.node_id for n in knowledge_graph_nodes}

        # Create a combined dictionary of nodes with scores
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in knowledge_graph_nodes})

        # Perform set operation
        if self._mode == constants.BOOLEAN_AND:
            retrieve_ids = vector_ids.intersection(knowledge_graph_ids)
        elif self._mode == constants.BOOLEAN_OR:
            retrieve_ids = vector_ids.union(knowledge_graph_ids)
        else:
            raise ValueError(
                f"""Set operation is not defined for invalid retriever logical combination mode {self._mode}, 
                which must be either {constants.BOOLEAN_AND} (intersection) or {constants.BOOLEAN_OR} (union)."""
            )

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
