import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
MAX_BODY_CHARS = 2000
MAX_COMMENT_CHARS = 4000
REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class GitHubClientError(Exception):
    """Custom exception that guarantees no secrets or raw headers are leaked."""

    pass


class GitHubClient:
    def __init__(self, token: str | None = None, allowlist: list[str] | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise GitHubClientError("GITHUB_TOKEN is missing. Set it in .env or pass directly.")

        # Read allowlist from parameter or .env (comma-separated if multiple)
        env_allowlist = os.getenv("GITHUB_REPO_ALLOWLIST", "")
        self.allowlist = (
            allowlist
            if allowlist is not None
            else [r.strip() for r in env_allowlist.split(",") if r.strip()]
        )

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "mcp-assistant-app",
        }

    def _validate_repo(self, repo: str) -> str:
        """Enforce strict 'owner/repo' format, strip URL prefixes, and check allowlist."""
        if not isinstance(repo, str):
            raise GitHubClientError(f"Invalid repo format '{repo}'. Expected format 'owner/repo'.")

        clean = repo.strip()
        # Automatically clean URL prefixes if user or LLM passes full URL
        if "github.com/" in clean:
            clean = clean.split("github.com/")[-1].strip().strip("/")

        if not REPO_PATTERN.match(clean):
            raise GitHubClientError(f"Invalid repo format '{repo}'. Expected format 'owner/repo'.")

        allowlist_lower = [r.lower() for r in self.allowlist]
        if self.allowlist and clean.lower() not in allowlist_lower:
            raise GitHubClientError(
                f"Access denied: Repo '{clean}' is not in allowlist ({self.allowlist})."
            )
        return clean

    def _validate_issue_number(self, issue_number: int) -> None:
        """Ensure issue number is a strictly positive integer."""
        if not isinstance(issue_number, int) or issue_number <= 0:
            raise GitHubClientError(
                f"Invalid issue number '{issue_number}'. Must be a positive integer."
            )

    def _safe_request(self, method: str, url: str, json_data: dict[str, Any] | None = None) -> Any:
        """Execute HTTP request and catch errors, ensuring secrets never leak in messages."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.request(method, url, headers=self.headers, json=json_data)
                if response.status_code == 404:
                    raise GitHubClientError(
                        "Resource not found (404). Check repo name or issue number."
                    )
                if response.status_code == 401:
                    raise GitHubClientError(
                        "Authentication failed (401). Token is invalid or expired."
                    )
                if response.status_code == 403:
                    raise GitHubClientError(
                        "Permission denied (403). Token lacks required permissions."
                    )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise GitHubClientError("GitHub API request timed out after 10 seconds.") from None
        except httpx.HTTPStatusError as e:
            raise GitHubClientError(
                f"GitHub API returned error status: {e.response.status_code}"
            ) from None
        except httpx.RequestError:
            raise GitHubClientError("Network error while connecting to GitHub API.") from None

    def list_issues(self, repo: str) -> list[dict[str, Any]]:
        """List issues for an allowed repo, returning clean, truncated data."""
        clean_repo = self._validate_repo(repo)
        url = f"{GITHUB_API_BASE}/repos/{clean_repo}/issues"
        raw_issues = self._safe_request("GET", url)

        cleaned_issues = []
        for issue in raw_issues:
            if "pull_request" in issue:
                continue

            body = (issue.get("body") or "").strip()
            if len(body) > MAX_BODY_CHARS:
                body = body[:MAX_BODY_CHARS] + "... [truncated for context length]"

            cleaned_issues.append(
                {
                    "number": issue.get("number"),
                    "title": issue.get("title", ""),
                    "state": issue.get("state", ""),
                    "author": issue.get("user", {}).get("login", "unknown"),
                    "body": body,
                }
            )
        return cleaned_issues

    def get_issue(self, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch a single issue and its comments by number with strict validation."""
        clean_repo = self._validate_repo(repo)
        self._validate_issue_number(issue_number)

        url = f"{GITHUB_API_BASE}/repos/{clean_repo}/issues/{issue_number}"
        issue = self._safe_request("GET", url)

        # Also fetch any comments on this issue
        comments_url = f"{GITHUB_API_BASE}/repos/{clean_repo}/issues/{issue_number}/comments"
        try:
            raw_comments = self._safe_request("GET", comments_url)
            comments = [
                {
                    "author": c.get("user", {}).get("login", "unknown"),
                    "body": (c.get("body") or "")[:500],
                }
                for c in (raw_comments if isinstance(raw_comments, list) else [])
            ]
        except Exception:
            comments = []

        body = (issue.get("body") or "").strip()
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "... [truncated for context length]"

        return {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "state": issue.get("state", ""),
            "author": issue.get("user", {}).get("login", "unknown"),
            "body": body,
            "comments": comments,
        }

    def create_comment(self, repo: str, issue_number: int, comment: str) -> dict[str, str]:
        """Add a comment with length limits to prevent spam/abuse."""
        clean_repo = self._validate_repo(repo)
        self._validate_issue_number(issue_number)

        if not comment or not comment.strip():
            raise GitHubClientError("Comment cannot be empty.")

        clean_comment = comment.strip()
        if len(clean_comment) > MAX_COMMENT_CHARS:
            raise GitHubClientError(
                f"Comment exceeds max allowed length of {MAX_COMMENT_CHARS} characters."
            )

        url = f"{GITHUB_API_BASE}/repos/{clean_repo}/issues/{issue_number}/comments"
        data = self._safe_request("POST", url, json_data={"body": clean_comment})

        return {"comment_url": data.get("html_url", "")}
