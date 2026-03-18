"""Lightweight local retrieval for safety grounding.

This module provides a dependency-light retrieval layer over local markdown/text
files. It is a practical scaffold for future migration to a vector database.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "")}


def _chunk_text(text: str) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
    return chunks or ([text.strip()] if text.strip() else [])


def _candidate_files(docs_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for pattern in ("*.md", "*.txt"):
        candidates.extend(sorted(docs_dir.rglob(pattern)))
    return [path for path in candidates if path.is_file()]


def retrieve_relevant_context(
    query: str,
    docs_dir: Optional[str | Path] = None,
    max_chunks: Optional[int] = None,
) -> Dict:
    if not settings.RETRIEVAL_ENABLED:
        return {
            "context": "",
            "sources": [],
            "retrieval_enabled": False,
        }

    resolved_docs_dir = Path(docs_dir or settings.RETRIEVAL_DOCS_DIR).resolve()
    if not resolved_docs_dir.exists():
        return {
            "context": "",
            "sources": [],
            "retrieval_enabled": True,
            "retrieval_warning": f"docs directory not found: {resolved_docs_dir}",
        }

    query_terms = _tokenize(query)
    if not query_terms:
        return {
            "context": "",
            "sources": [],
            "retrieval_enabled": True,
        }

    scored: List[Dict] = []
    for path in _candidate_files(resolved_docs_dir):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for chunk in _chunk_text(raw):
            chunk_terms = _tokenize(chunk)
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            scored.append(
                {
                    "path": str(path),
                    "score": round(score, 4),
                    "excerpt": chunk[:600],
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    top_n = max_chunks or settings.RETRIEVAL_MAX_CHUNKS
    top = scored[:top_n]

    context_parts = []
    for item in top:
        source_name = Path(item["path"]).name
        context_parts.append(f"Source: {source_name}\n{item['excerpt']}")

    return {
        "context": "\n\n".join(context_parts),
        "sources": top,
        "retrieval_enabled": True,
    }
