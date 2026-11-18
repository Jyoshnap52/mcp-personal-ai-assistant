import httpx
import pytest

from mcp_servers.github.client import (
    MAX_BODY_CHARS,
    MAX_COMMENT_CHARS,
    GitHubClient,
    GitHubClientError,
)


@pytest.fixture
def client():
    """Create a test client with an isolated allowlist."""
    return GitHubClient(
        token="test_mock_token_123",
        allowlist=["allowed-owner/allowed-repo"],
    )


# --- 1. Allowlist & Format Validation Tests ---


def test_disallowed_repo_rejected(client):
    """Ensure accessing a non-allowlisted repo raises GitHubClientError."""
    with pytest.raises(GitHubClientError, match="not in allowlist"):
        client.list_issues("evil-hacker/stolen-repo")


def test_invalid_repo_format(client):
    """Ensure malformed repo strings are rejected immediately."""
    with pytest.raises(GitHubClientError, match="Invalid repo format"):
        client.list_issues("invalid_format_without_slash")


def test_invalid_issue_number(client):
    """Ensure non-positive issue numbers are rejected."""
    with pytest.raises(GitHubClientError, match="Must be a positive integer"):
        client.get_issue("allowed-owner/allowed-repo", -1)

    with pytest.raises(GitHubClientError, match="Must be a positive integer"):
        client.get_issue("allowed-owner/allowed-repo", 0)


# --- 2. Body Truncation & Field Filtering Tests ---


def test_list_issues_truncates_long_body_and_filters_prs(client, monkeypatch):
    """Ensure long issue bodies are truncated and PRs are excluded."""
    giant_body = "x" * (MAX_BODY_CHARS + 500)

    fake_api_response = [
        {
            "number": 1,
            "title": "Bug with long logs",
            "state": "open",
            "user": {"login": "tester"},
            "body": giant_body,
        },
        {
            "number": 2,
            "title": "A Pull Request",
            "state": "open",
            "pull_request": {"url": "https://api.github.com/..."},
            "user": {"login": "coder"},
            "body": "PR description",
        },
    ]

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_api_response

        def raise_for_status(self):
            pass

    # Mock httpx.Client.request to return our FakeResponse
    monkeypatch.setattr(httpx.Client, "request", lambda *args, **kwargs: FakeResponse())

    issues = client.list_issues("allowed-owner/allowed-repo")

    # Only 1 issue returned (PR was skipped)
    assert len(issues) == 1
    assert issues[0]["number"] == 1
    assert issues[0]["author"] == "tester"
    # Body must be truncated
    assert len(issues[0]["body"]) < len(giant_body)
    assert "[truncated for context length]" in issues[0]["body"]


# --- 3. Comment Validation & Error Handling Tests ---


def test_comment_too_long_rejected(client):
    """Ensure spam comments exceeding MAX_COMMENT_CHARS are blocked."""
    giant_comment = "spam " * (MAX_COMMENT_CHARS // 4 + 10)
    with pytest.raises(GitHubClientError, match="exceeds max allowed length"):
        client.create_comment("allowed-owner/allowed-repo", 1, giant_comment)


def test_empty_comment_rejected(client):
    """Ensure blank comments are rejected."""
    with pytest.raises(GitHubClientError, match="Comment cannot be empty"):
        client.create_comment("allowed-owner/allowed-repo", 1, "   ")


def test_http_404_maps_to_clean_error(client, monkeypatch):
    """Ensure 404 responses turn into clean errors without leaking headers."""

    class Fake404Response:
        status_code = 404

        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx.Client, "request", lambda *args, **kwargs: Fake404Response())

    with pytest.raises(GitHubClientError, match="Resource not found"):
        client.get_issue("allowed-owner/allowed-repo", 9999)
