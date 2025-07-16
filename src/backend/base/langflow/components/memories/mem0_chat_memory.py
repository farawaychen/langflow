import asyncio
from langflow.schema.message import Message
from loguru import logger
from langchain_core.messages import BaseMessage

from langflow.base.memory.mem0 import Mem0LCMemory
from langflow.base.memory.model import LCChatMemoryComponent
from langflow.inputs import DictInput, HandleInput, MessageTextInput, NestedDictInput, SecretStrInput
from langflow.io import Output
from langflow.schema import Data

class Mem0MemoryComponent(LCChatMemoryComponent):
    display_name = "Mem0 Chat Memory"
    description = "Retrieves and stores user memory."
    name = "mem0_chat_memory"
    icon: str = "Mem0"
    inputs = [
        NestedDictInput(
            name="mem0_config",
            display_name="Mem0 Configuration",
            info="""Configuration dictionary for initializing Mem0 memory instance.
                    Example:
                    {
                        "graph_store": {
                            "provider": "neo4j",
                            "config": {
                                "url": "neo4j+s://your-neo4j-url",
                                "username": "neo4j",
                                "password": "your-password"
                            }
                        },
                        "version": "v1.1"
                    }""",
            input_types=["Data"],
        ),
        MessageTextInput(
            name="ingest_message",
            display_name="Message to Ingest",
            info="The message content to be ingested into Mem0 memory.",
        ),
        HandleInput(
            name="existing_memory",
            display_name="Existing Memory Instance",
            input_types=["Memory"],
            info="Optional existing Mem0 memory instance. If not provided, a new instance will be created.",
        ),
        MessageTextInput(
            name="user_id", 
            display_name="User ID", 
            info="Identifier for the user associated with the messages."
        ),
        MessageTextInput(
            name="search_query", 
            display_name="Search Query", 
            info="Input text for searching related memories.",
            tool_mode=True,
            required=True,
        ),
        SecretStrInput(
            name="mem0_api_key",
            display_name="Mem0 API Key",
            info="API key for Mem0 platform. Leave empty to use the local version.",
        ),
        DictInput(
            name="metadata",
            display_name="Metadata",
            info="Additional metadata to associate with the ingested message.",
            advanced=True,
        ),
        SecretStrInput(
            name="openai_api_key",
            display_name="OpenAI API Key",
            required=False,
            info="API key for OpenAI. Required if using OpenAI Embeddings without a provided configuration.",
        ),
    ]

    outputs = [
        Output(name="memory", display_name="Mem0 Memory", method="build_memory"),
        Output(name="search_results", display_name="Search Results", method="build_search_results"),
    ]

    async def build_memory(self) -> Mem0LCMemory:
        """Build or return an existing Mem0 memory instance."""
        if not self.user_id:
            msg = "user_id is required"
            raise ValueError(msg)

        if self.existing_memory and isinstance(self.existing_memory, Mem0LCMemory):
            logger.info("Using existing Mem0 memory instance")
            return self.existing_memory

        memory = Mem0LCMemory(
            user_id=self.user_id,
            api_key=self.mem0_api_key,
            config=self.mem0_config,
        )

        if self.search_query:
            from langchain_core.messages import HumanMessage
            
            asyncio.create_task(memory.aadd_messages(HumanMessage(content=self.search_query)))

        return memory

    async def build_search_results(self) -> list[Message]:
        """Search for related memories using the provided query."""
        memory = await self.build_memory()

        if not self.search_query:
            logger.warning("No search query provided")
            return []

        try:
            # Use the underlying Mem0 client for search
            memorys = await memory.aget_messages_from_mem0(
                query=self.search_query
            )
            logger.info(f"Found {len(memorys)} related memories")
            logger.info(f"Ori Related memories retrieved: {memorys}")
            memorys = [Message.from_lc_message(mem) for mem in memorys]
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
        return memorys
