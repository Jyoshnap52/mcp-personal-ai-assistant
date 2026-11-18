import json

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    from mcp.server.mcpserver import MCPServer

from mcp_servers.github.client import GitHubClient, GitHubClientError

# Initialize MCP 2.x Server
server = MCPServer("github")
client = GitHubClient()


@server.tool()
def list_issues(repo: str) -> str:
    """List issues for an allowed repository, returning number, title, author, and body."""
    try:
        issues = client.list_issues(repo)
        return json.dumps(issues, indent=2)
    except GitHubClientError as e:
        return f"Error: {e}"


@server.tool()
def get_issue(repo: str, issue_number: int) -> str:
    """Fetch details for a single issue by its positive issue number."""
    try:
        issue = client.get_issue(repo, issue_number)
        return json.dumps(issue, indent=2)
    except GitHubClientError as e:
        return f"Error: {e}"


@server.tool()
def create_comment(repo: str, issue_number: int, comment: str) -> str:
    """Add a comment to an issue in an allowed repository and return its link."""
    try:
        result = client.create_comment(repo, issue_number, comment)
        return json.dumps(result, indent=2)
    except GitHubClientError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    server.run("stdio")
