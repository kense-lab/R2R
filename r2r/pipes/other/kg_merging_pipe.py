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
from r2r.base.abstractions.llm import GenerationConfig
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

    async def deduplicate_entities(self, entities: list[EntityNode]) -> None:
        """
        Adds descriptions to the entities in the knowledge graph.
        """

        entity_names = [entity.name for entity in entities]

        triplets = self.kg_provider.get_triplets(entities = entity_names)

        fragment_ids_dict = {}
        for row in triplets: 
            if row['source_id'] not in fragment_ids_dict:
                fragment_ids_dict[row['source_id']] = []
            fragment_ids_dict[row['source_id']].append(row['source_properties']['fragment_ids']) # or source_label_properties
        
        # now db query
        flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

        entity_descriptions = {}
        for name, fragment_ids in fragment_ids_dict:

            fragment_ids = list(set(flatten(fragment_ids)))

            fragments = self.database_provider.get_fragments_by_id(fragment_ids)

            fragments_merged = ""
            for i, fragment in enumerate(fragments):
                fragments_merged += 'Chunk {i}:'
                fragments_merged += fragment
                fragments_merged += '\n'

            messages = [{'role': 'user', 'content': 
                            f"""You are given en entity and a list of chunks that describe the entity. Give me a short summary of the entity based on the description in the chunk. 
                                Entity:\n{name}
                                Chunks:\n{fragments_merged}
                            """
                         }]

            description = self.llm_provider.get_completion(messages = messages, generation_config= GenerationConfig(model='gpt-4o-mini'))
            # embedding 
            embedding = self.embedding_provider.get_embedding(description)
            entity_descriptions[name] = {'description': description, 'embedding': embedding}


        append_entity_descriptions = self.kg_provider.add_descriptions(entity_descriptions)

        all_entity_pairs = []
        for name, fragment_ids in fragment_ids_dict:
            # find 5 closest nodes in terms of L2 distance with description
            closest_nodes = self.kg_provider.get_closest_nodes(name)

            messages = [{'role': 'user', 'content': 
                f"""You are given en entity and a list of chunks that describe the entity. Give me a short summary of the entity based on the description in the chunk. 
                    Entity:\n{name}
                    Chunks:\n{fragments_merged}
                """
                }]

            pairs_of_entities = self.llm_provider.get_completion(messages=messages, generation_config=GenerationConfig(model='gpt-4o-mini'))    
            # make it so that we get a tuple of entities.
            all_entity_pairs.extend(pairs_of_entities)

            # get cycles?
        final_replacement_pairs = self.get_replacement_pairs(all_entity_pairs)

        # merge_entities
        merged_entities = self.kg_provider.merge_pairs_of_entities(final_replacement_pairs)

        return merged_entities

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

        entities = self.kg_provider.get_nodes()
        merged_entities = self.deduplicate_entities(entities)
        yield None
