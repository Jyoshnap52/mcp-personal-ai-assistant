import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


async def main():
    repo = os.getenv("GITHUB_REPO_ALLOWLIST")
    print("🚀 Spawning GitHub MCP Server over stdio...")

    # Parameters to launch the server using the current Python environment
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.github.server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize MCP handshake
            await session.initialize()
            print("🤝 MCP Handshake successful!")

            # 2. Ask the server what tools it offers
            tools_response = await session.list_tools()
            print(f"\n📋 Registered MCP Tools ({len(tools_response.tools)}):")
            for tool in tools_response.tools:
                print(f"  • {tool.name}: {tool.description}")

            # 3. Call 'list_issues' through MCP!
            print(f"\n🎯 Calling 'list_issues' via MCP for repo: {repo} ...")
            result = await session.call_tool(
                "list_issues",
                arguments={"repo": repo},
            )
            print("\n📬 Result received from MCP Server:")
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
