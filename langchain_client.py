import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    temperature=0,
    
)


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [
                    "D:/F_drive/Nikhil/dev/udemy/mcp-crash-course/servers/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )
    async with client:
        tools = client.get_tools()
        print("Loaded tools:", [tool.name for tool in tools])
        agent = create_react_agent(
            llm,
            tools,
            prompt=(
                "You are a helpful AI assistant that answers questions related to weather and calculations using the available MCP tools. "
                
            ),
        )
        result = await agent.ainvoke(
            {
                "messages": "What is the weather in San Francisco? Extract the temperature and triple it."
            }
        )

        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())