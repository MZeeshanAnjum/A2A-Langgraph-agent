from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from langchain_core.messages import HumanMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
import uuid
from a2a.types import Message, Part, TextPart

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from a2a.utils.errors import ServerError
from google.adk.a2a.utils.agent_to_a2a import to_a2a

class State(BaseModel):
    query: str
    response: str = ""

def llm_node(state: State):
    return {"response": f"The current weather in Lahore is very hot. It is around 40 degrees Celsius."}

builder = StateGraph(State)
builder.add_node("Chatbot", llm_node)
builder.add_edge(START, "Chatbot")
builder.add_edge("Chatbot", END)
graph = builder.compile()

def invoke(query: str) -> str:
    initial_state = {"query": query}
    result = graph.invoke(initial_state)
    return result["response"]


class MyLangGraphAgent(AgentExecutor):
    # Give your agent a name and description for ADK
    name = "MyLangGraphAgent"
    description = "An agent that uses a language graph for weather updates."
    sub_agents = []
    tools = []

    def __init__(self):
        self.agent = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        query = context.get_user_input()
        task = context.current_task

        result = invoke(query)
        # build the Message explicitly

        parts=[Part(root=TextPart(text=result))]
        # enqueue the response message
        await updater.add_artifact(parts)

        # mark task as complete
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


agent_skill = AgentSkill(
    id="weather_info",
    name="MyLangGraphAgent",
    description="An agent that uses a language graph for weather updates.",
    tags=["weather", "information"],
)
agent_card = AgentCard(
    name="MyLangGraphAgent",
    description="An agent that uses a language graph for weather updates.",
    url = "http://localhost:8001/invoke",
    skills=[agent_skill],
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=True
    ),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    version="1.0.0",
)

httpx_client = httpx.AsyncClient()
request_handler = DefaultRequestHandler(
    agent_executor=MyLangGraphAgent(),
    task_store=InMemoryTaskStore(),
)
server = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

app = server.build()  # this is your Starlette ASGI app

print("Listing routes:")
for route in app.routes:
    print(f"Path: {route.path}, name: {getattr(route, 'name', None)}, methods: {getattr(route, 'methods', None)}")

uvicorn.run(app, host="0.0.0.0", port=8001)


#  Invoke-RestMethod -Uri "http://localhost:8001/" `
# >>   -Method POST `
# >>   -ContentType "application/json" `
# >>   -Body '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"conversation":"conv-1","message":{"messageId":"msg-123","role":"user","parts":[{"contentType":"text/plain","text":"Hello"}]}}}' `
# >> | ConvertTo-Json -Depth 10
# >>
# {
#     "id":  "1",
#     "jsonrpc":  "2.0",
#     "result":  {
#                    "kind":  "message",
#                    "messageId":  "msg-c097fc8f-2457-4273-b979-cd853fa1ed55",
#                    "parts":  [
#                                  {
#                                      "kind":  "text",
#                                      "text":  "The current weather in Lahore is very hot. (you asked: Hello)"
#                                  }
#                              ],
#                    "role":  "agent"
#                }
# }