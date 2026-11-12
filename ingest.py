"""Document loading, chunking and hybrid (dense + BM25) retrieval. In-memory, per session."""
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def load_file(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Return (filename, [(page_number, text), ...])."""
    name = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return name, [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
    with open(path, encoding="utf-8", errors="ignore") as f:
        return name, [(1, f.read())]


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def _tok(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


class Index:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.emb = None
        self.bm25 = None

    def add_files(self, paths: List[str]) -> int:
        new: List[Chunk] = []
        for p in paths:
            name, pages = load_file(p)
            for page, txt in pages:
                for c in chunk_text(txt):
                    new.append(Chunk(c, name, page))
        if not new:
            return 0
        vecs = get_embedder().encode([c.text for c in new], normalize_embeddings=True)
        self.emb = vecs if self.emb is None else np.vstack([self.emb, vecs])
        self.chunks += new
        self.bm25 = BM25Okapi([_tok(c.text) for c in self.chunks])
        return len(new)

    def search(self, query: str, k: int = 6) -> List[Chunk]:
        """Hybrid search: reciprocal rank fusion of dense and BM25 rankings."""
        if not self.chunks:
            return []
        q = get_embedder().encode([query], normalize_embeddings=True)[0]
        dense_rank = np.argsort(-(self.emb @ q))
        bm_rank = np.argsort(-self.bm25.get_scores(_tok(query)))
        fused = {}
        for ranking in (dense_rank, bm_rank):
            for r, idx in enumerate(ranking[: 4 * k]):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + r)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in top]
