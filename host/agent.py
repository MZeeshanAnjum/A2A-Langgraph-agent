import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from a2a.types import Message, Part, TextPart
from remote_agent_connection import RemoteAgentConnections

# remote_agent_connections = {}
# cards = {}

load_dotenv()
nest_asyncio.apply()

# async def _async_init_components(remote_agent_addresses: List[str]):
#     async with httpx.AsyncClient(timeout=30) as client:
#         for address in remote_agent_addresses:
#             card_resolver = A2ACardResolver(client, address)
#             try:
#                 card = await card_resolver.get_agent_card()
#                 remote_connection = RemoteAgentConnections(
#                     agent_card=card, agent_url=address
#                 )
#                 remote_agent_connections[card.name] = remote_connection
#                 cards[card.name] = card
#             except httpx.ConnectError as e:
#                 print(f"ERROR: Failed to get agent card from {address}: {e}")
#             except Exception as e:
#                 print(f"ERROR: Failed to initialize connection for {address}: {e}")

#     agent_info = [
#         json.dumps({"name": card.name, "description": card.description})
#         for card in cards.values()
#     ]
#     print("agent_info:", agent_info)
#     agents = "\n".join(agent_info) if agent_info else "No friends found"
#     print(f"Agents in the network:\n{agents}")
#     return 
# asyncio.run(_async_init_components(["http://localhost:8000"]))

class HostAgent:
    """The Host agent."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.0-flash",
            name="Host_Agent",
            instruction=f"You are a helpful agent, You have access to these agents. You can use send_message tool to send a message to the agent \n {self.agents}",
            description="This Host agent orchestrates scheduling pickleball with friends.",
            tools=[
                self.send_message
            ],
        )

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to a remote friend agent."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        message_id = str(uuid.uuid4())
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return {"error": "remote agent did not return a valid Task"}

        # Clean result
        task_obj = send_response.root.result
        # Extract artifacts/parts if available
        response_content = task_obj.model_dump_json()
        print(response_content)
        try:
            json_content = json.loads(response_content)
        except Exception:
            return {"remote_task_id": task_obj.id, "status": task_obj.status.state, "parts": []}

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
                    print("Extracted parts:", artifact["parts"])

        msg = Message(
            messageId=str(uuid.uuid4()),
            role="agent",
            parts=[Part(root=TextPart(text="The current weather in Lahore is very hot. It is around 40°C"))]
        )
        return msg

def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # Hardcoded URLs for the friend agents
        friend_agent_urls = ["http://localhost:8001"]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync()
