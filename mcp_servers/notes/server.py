import json
import re
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# Define data directory relative to project root
NOTES_DIR = Path(__file__).resolve().parents[2] / "notes_data"

server = MCPServer("notes")


class NoteSearchResult(BaseModel):
    filename: str
    score: float
    text: str


class NotesIndex:
    """In-memory BM25 index built once at server startup."""

    def __init__(self, notes_dir: Path):
        self.notes_dir = notes_dir
        self.filenames: list[str] = []
        self.documents: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._load_and_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _load_and_index(self) -> None:
        if not self.notes_dir.exists():
            return

        for filepath in sorted(self.notes_dir.glob("*.md")):
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                self.filenames.append(filepath.name)
                self.documents.append(content)
                self.tokenized_corpus.append(self._tokenize(content))

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 3) -> list[NoteSearchResult]:
        if not self.bm25 or not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.filenames, scores, self.documents, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for filename, score, doc in ranked[:k]:
            if score > 0:
                results.append(
                    NoteSearchResult(
                        filename=filename,
                        score=round(float(score), 4),
                        text=doc,
                    )
                )
        return results


# Load once at startup
index = NotesIndex(NOTES_DIR)


@server.tool()
def search_notes(query: str, k: int = 3) -> str:
    """Search internal notes and policies using BM25 keyword search. Returns top k matches."""
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    safe_k = max(1, min(k, 10))
    results = index.search(query.strip(), k=safe_k)
    if not results:
        return f"No matching notes found for query: '{query}'"

    return json.dumps([r.model_dump() for r in results], indent=2)


if __name__ == "__main__":
    server.run("stdio")
