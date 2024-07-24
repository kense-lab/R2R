# merging nodes that represent the same entity in the knowledge graph
# and updating the edges accordingly

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from r2r.base import (
    AsyncState,
    EmbeddingProvider,
    KGExtraction,
    KGProvider,
    KVLoggingSingleton,
    PipeType,
    LLMProvider,
    DatabaseProvider,
)
from r2r.base.abstractions.llama_abstractions import EntityNode, Relation
from r2r.base.pipes.base_pipe import AsyncPipe

logger = logging.getLogger(__name__)


class KGMergingePipe(AsyncPipe):
    class Input(AsyncPipe.Input):
        message: AsyncGenerator[KGExtraction, None]

    def __init__(
        self,
        kg_provider: KGProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        database_provider: Optional[DatabaseProvider] = None,
        LLMProvider: Optional[LLMProvider] = None,
        storage_batch_size: int = 100000,
        pipe_logger: Optional[KVLoggingSingleton] = None,
        type: PipeType = PipeType.INGESTOR,
        config: Optional[AsyncPipe.PipeConfig] = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the async knowledge graph storage pipe with necessary components and configurations.
        """
        logger.info(
            f"Initializing an `KGStoragePipe` to store knowledge graph extractions in a graph database."
        )

        super().__init__(
            pipe_logger=pipe_logger,
            type=type,
            config=config,
            *args,
            **kwargs,
        )
        self.kg_provider = kg_provider
        self.embedding_provider = embedding_provider
        self.database_provider = database_provider
        self.llm_provider = LLMProvider
        self.storage_batch_size = storage_batch_size

    # async def store(
    #     self,
    #     kg_extractions: list[KGExtraction],
    # ) -> list:
    #     """
    #     Stores a batch of knowledge graph extractions in the graph database.
    #     """
    #     try:
    #         nodes = []
    #         relations = []
    #         for extraction in kg_extractions:
    #             for entity in extraction.entities.values():
    #                 embedding = None
    #                 if self.embedding_provider:
    #                     embedding = self.embedding_provider.get_embedding(
    #                         "Entity:\n{entity.value}\nLabel:\n{entity.category}\nSubcategory:\n{entity.subcategory}"
    #                     )
    #                 nodes.append(
    #                     EntityNode(
    #                         name=entity.value,
    #                         label=entity.category,
    #                         embedding=embedding,
    #                         properties=(
    #                             {"subcategory": entity.subcategory, 'fragment_id': [str(extraction.fragment_id)]}
    #                             if entity.subcategory
    #                             else {'fragment_id': [str(extraction.fragment_id)]}
    #                         ),
    #                     )
    #                 )
    #             for triple in extraction.triples:
    #                 relations.append(
    #                     Relation(
    #                         source_id=triple.subject,
    #                         target_id=triple.object,
    #                         label=triple.predicate,
    #                         properties={"fragment_id": [str(extraction.fragment_id)]},
    #                     )
    #                 )
    #         nodes = self.kg_provider.upsert_nodes(nodes)
    #         self.kg_provider.upsert_relations(relations)

    #         return nodes   

    #     except Exception as e:
    #         error_message = f"Failed to store knowledge graph extractions in the database: {e}"
    #         logger.error(error_message)
    #         raise ValueError(error_message)

    async def add_descriptions(self, entities: list[EntityNode]) -> None:
        """
        Adds descriptions to the entities in the knowledge graph.
        """
        for entity 


    async def _run_logic(
        self,
        input: Input,
        state: AsyncState,
        run_id: uuid.UUID,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[None, None]:
        """
        Executes the async knowledge graph storage pipe: storing knowledge graph extractions in the graph database.
        """
        batch_tasks = []
        kg_batch = []


        # get all graph nodes
        nodes = self.kg_provider.get_nodes()

        node_descriptions = self.database_provider.get_node_descriptions()


        await asyncio.gather(*batch_tasks)

        yield None
