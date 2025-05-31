from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
import asyncio
from dotenv import load_dotenv
load_dotenv()
# Debug mode flag - set to False to disable debug prints
DEBUG = False

# Initialize LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4.1-nano", streaming=True, max_tokens=250)

# Set up the prompt template
system_prompt = """You are a friendly female and highly efficient hospital assistant. Your primary job is to help patients book doctor appointments and handle medical emergencies. You should:

1. For appointments:
- Help gather patient specialty needs
- Assist with scheduling
- Collect patient details efficiently
- Be friendly and professional

2. For emergencies:
- Respond with urgency and clarity
- Gather location quickly
- Provide basic emergency guidance
- Stay reassuring and focused

Please be friendly, polite, and efficient for appointments, but switch to an urgent, clear, and reassuring tone for emergencies.

Respond in Hindi when users speak Hindi, otherwise use English."""
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

# Create the pipeline
pipeline = prompt_template | llm

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        if DEBUG:
            print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

# Chat history management
chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    if DEBUG:
        print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    return chat_map[session_id]

# Create the pipeline with history
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

# Function to generate stream from a query
async def generate_stream(query, session_id="default", history_size=4):
    """
    Generate a stream response for a given query.
    
    Args:
        query (str): The query text to send to the LLM
        session_id (str): Session ID for maintaining conversation history
        history_size (int): Number of messages to keep in history
        
    Returns:
        AsyncGenerator: Stream of response chunks
    """
    return pipeline_with_history.astream(
        {"query": query},
        config={"session_id": session_id, "k": history_size}
    )

# Example usage function
async def print_stream(query="What can you help me with?", session_id="default", history_size=4):
    """Print the streaming response to console."""
    stream = await generate_stream(query, session_id, history_size)
    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # Add newline at the end

# Only run this if the file is executed directly (not imported)
if __name__ == "__main__":
    # Enable debug prints for direct execution
    DEBUG = True
    asyncio.run(print_stream(
        "what is my name of india again? and details in 500 words", 
        session_id="id_k14", 
        history_size=14
    ))