import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


async def run_dual_mcp_client():
    repo = os.getenv("GITHUB_REPO_ALLOWLIST")
    python_bin = sys.executable

    print("=" * 60)
    print("🤖 LAUNCHING DUAL MCP SERVERS OVER STDIO...")
    print("=" * 60)

    # 1. Configure server parameters
    github_params = StdioServerParameters(
        command=python_bin,
        args=["-m", "mcp_servers.github.server"],
        env=dict(os.environ),
    )
    notes_params = StdioServerParameters(
        command=python_bin,
        args=["-m", "mcp_servers.notes.server"],
        env=dict(os.environ),
    )

    # 2. Connect to both servers concurrently
    async with (
        stdio_client(github_params) as (gh_read, gh_write),
        stdio_client(notes_params) as (notes_read, notes_write),
    ):
        async with (
            ClientSession(gh_read, gh_write) as gh_session,
            ClientSession(notes_read, notes_write) as notes_session,
        ):
            # Handshakes
            await gh_session.initialize()
            await notes_session.initialize()
            print("✅ Successfully handshaked with both GitHub & Notes MCP servers!\n")

            # 3. List all registered tools from both servers
            gh_tools = await gh_session.list_tools()
            notes_tools = await notes_session.list_tools()

            print("📦 [GitHub Server Tools]:")
            for t in gh_tools.tools:
                print(f"  • {t.name}: {t.description}")

            print("\n📦 [Notes Server Tools]:")
            for t in notes_tools.tools:
                print(f"  • {t.name}: {t.description}")

            print("\n" + "=" * 60)
            print("🧪 TESTING ONE TOOL FROM EACH SERVER:")
            print("=" * 60)

            # 4. Call Notes tool: search_notes
            query = "how many sick days do we get?"
            print(f"\n1️⃣ Calling Notes MCP: search_notes(query='{query}', k=1)...")
            note_result = await notes_session.call_tool(
                "search_notes",
                arguments={"query": query, "k": 1},
            )
            print("📄 Notes Search Result:")
            for content in note_result.content:
                print(content.text)

            # 5. Call GitHub tool: list_issues
            print(f"\n2️⃣ Calling GitHub MCP: list_issues(repo='{repo}')...")
            gh_result = await gh_session.call_tool(
                "list_issues",
                arguments={"repo": repo},
            )
            print("🐙 GitHub Issues Result:")
            for content in gh_result.content:
                print(content.text)

            print("\n" + "=" * 60)
            print("🎉 DUAL MCP INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_dual_mcp_client())
