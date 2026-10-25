"""Retrieval index over ``docs/corpus`` (``make index``).

Chunking: one chunk per ``##`` section, split further on paragraph boundaries when a section is
longer than ``assistant.chunk_max_words``. Every chunk records its document, title, section,
language and source file. Chunks are embedded with a *multilingual* sentence model so that a French
question lands on an English policy chunk (cosine similarity on L2-normalised vectors).

The index is a static artifact (``artifacts/index/``): ``chunks.json``, ``embeddings.npy`` and
``index_meta.json`` (model, corpus hash, build time). Building is offline; the API only embeds the
query and takes a dot product — at ~100 chunks a flat search is faster than any ANN structure.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.utils import read_json, utc_now, write_json


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    lang: str
    source: str
    text: str


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """ONNX sentence embeddings via fastembed (no torch: keeps the serving image small)."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        cache = cache_dir or os.environ.get("BOREAL_EMBED_CACHE") or ".cache/fastembed"
        self._model = TextEmbedding(model_name, cache_dir=cache)

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalise(np.asarray(list(self._model.embed(texts)), dtype=np.float32))


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and CI (no model download).

    It is monolingual and lexical, so it is *not* a stand-in for retrieval quality — only for the
    plumbing around it.
    """

    name = "hash-bow-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-zà-ÿ0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return normalise(out)


def normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(v / np.where(n == 0, 1, n), dtype=np.float32)


def make_embedder(params: Params) -> Embedder:
    if os.environ.get("BOREAL_EMBEDDER") == "hash":
        return HashEmbedder()
    return FastEmbedder(params.assistant.embedding_model)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def parse_document(path: Path, max_words: int) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    if text.startswith("---"):
        _, front, text = text.split("---", 2)
        for line in front.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    doc_id = meta.get("doc_id", path.stem)
    title = meta.get("title", doc_id)
    lang = "en" if meta.get("lang", "en").startswith("en") else meta["lang"]
    chunks: list[Chunk] = []
    section, buf = "", list[str]()

    def flush() -> None:
        paras = [p.strip() for p in "\n".join(buf).split("\n\n") if p.strip()]
        paras = [p for p in paras if not p.startswith(">")]  # drop the pedagogical banner
        window: list[str] = []
        for p in paras:
            if window and sum(len(w.split()) for w in window) + len(p.split()) > max_words:
                _emit(window)
                window = []
            window.append(p)
        if window:
            _emit(window)

    def _emit(paras: list[str]) -> None:
        n = sum(c.section == section for c in chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{_slug(section) or 'intro'}-{n}",
                doc_id=doc_id,
                title=title,
                section=section or title,
                lang=lang,
                source=path.as_posix(),
                text="\n\n".join(paras),
            )
        )

    for line in text.splitlines():
        if line.startswith("# "):
            flush()
            buf, section = [], ""
            # A second top-level heading in a bilingual document starts its French half.
            if any(c.doc_id == doc_id for c in chunks) and meta.get("lang") == "en+fr":
                lang = "fr"
        elif line.startswith("## "):
            flush()
            buf, section = [], line[3:].strip()
        else:
            buf.append(line)
    flush()
    return [c for c in chunks if c.section != title or len(c.text.split()) > 15]


def load_corpus(corpus_dir: Path, max_words: int) -> list[Chunk]:
    return [c for p in sorted(corpus_dir.glob("*.md")) for c in parse_document(p, max_words)]


def corpus_sha256(corpus_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(corpus_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def embed_text(c: Chunk) -> str:
    return f"{c.title} — {c.section}\n{c.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def from_corpus(cls, corpus_dir: Path, embedder: Embedder, max_words: int) -> Retriever:
        chunks = load_corpus(corpus_dir, max_words)
        return cls(chunks, embedder.embed([embed_text(c) for c in chunks]), embedder)

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> Retriever:
        meta = read_json(index_dir / "index_meta.json")
        if meta["embedding_model"] != embedder.name:
            raise ValueError(
                f"index built with {meta['embedding_model']}, query embedder is {embedder.name}"
            )
        chunks = [Chunk(**c) for c in read_json(index_dir / "chunks.json")]
        return cls(chunks, np.load(index_dir / "embeddings.npy"), embedder)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ self.embedder.embed([query])[0]
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def load_retriever(params: Params, paths: Paths, embedder: Embedder | None = None) -> Retriever:
    """Serving-side loader: the committed index for the real embedder, the corpus otherwise."""
    embedder = embedder or make_embedder(params)
    if isinstance(embedder, FastEmbedder) and (paths.index_dir / "index_meta.json").exists():
        return Retriever.load(paths.index_dir, embedder)
    return Retriever.from_corpus(
        paths.root / params.assistant.corpus_dir, embedder, params.assistant.chunk_max_words
    )


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    corpus = paths.root / params.assistant.corpus_dir
    embedder = FastEmbedder(params.assistant.embedding_model)
    retriever = Retriever.from_corpus(corpus, embedder, params.assistant.chunk_max_words)
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.index_dir / "embeddings.npy", retriever.vectors)
    write_json(paths.index_dir / "chunks.json", [asdict(c) for c in retriever.chunks])
    meta = {
        "built_at": utc_now(),
        "embedding_model": embedder.name,
        "dim": int(retriever.vectors.shape[1]),
        "n_chunks": len(retriever.chunks),
        "n_documents": len({c.doc_id for c in retriever.chunks}),
        "chunk_max_words": params.assistant.chunk_max_words,
        "corpus_sha256": corpus_sha256(corpus),
        "chunks_per_document": {
            d: sum(c.doc_id == d for c in retriever.chunks)
            for d in sorted({c.doc_id for c in retriever.chunks})
        },
    }
    write_json(paths.index_dir / "index_meta.json", meta)
    print(
        f"index: {meta['n_chunks']} chunks from {meta['n_documents']} documents, dim {meta['dim']}"
    )
    return meta


if __name__ == "__main__":
    run()
