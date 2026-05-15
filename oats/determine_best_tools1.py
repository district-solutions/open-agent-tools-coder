#!/usr/bin/env python3
"""
Find best tool matches from a uses schema using BM25 + optional cross-encoder reranker.

1. Pipe schema from stdin (quickest)
cat schema.json | python3 determine_tool_matches.py -p "get date"

2. Pass schema file with -s
python3 determine_tool_matches.py -p "utc datetime" -s my_schema.json -k 5 -w 0.1

3. With reranker for higher precision
python3 determine_tool_matches.py -p "get date" -s schema.json -m bm25 -r
This runs BM25 first (fast), then the cross-encoder reranks the candidates for better semantic accuracy.

# Search all Coder Tools For the Best Match

```
./oats/determine_best_tools2.py -s ./.ai/AGENT.repo_uses.python.tools.json -p 'utc'
```
"""
import os
import sys
import argparse
import json
import traceback
from pathlib import Path
from typing import Any
from oats.log import gl

log = gl(__name__)


def load_schema(schema_path: str) -> dict[str, dict[str, str]]:
    """Load the tool-uses JSON schema from a file or stdin."""
    if schema_path == "-":
        data = json.load(sys.stdin)
    else:
        with Path(schema_path).open() as f:
            data = json.load(f)
    # Accept either {"uses": {...}} or the raw uses dict
    return data.get("uses", data)


def build_corpus(uses: dict[str, dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    """Build one document per (file, function) pair combining path + name + description."""
    corpus: list[str] = []
    meta: list[dict[str, str]] = []
    for file_path, functions in uses.items():
        for func_name, description in functions.items():
            readable = func_name.replace("_", " ")
            # doc = f"{file_path} {readable} {description}"
            doc = f"{func_name} {readable} {description}"
            corpus.append(doc)
            meta.append({"file": file_path, "func": func_name, "description": description, "doc": doc})
    return corpus, meta


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into whitespace-delimited tokens."""
    return text.lower().split()


def rank_with_bm25(
    query: str,
    corpus: list[str],
    meta: list[dict[str, str]],
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Rank corpus documents against a query using BM25 scoring."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(f"rank_bm25 not installed — run: pip install rank-bm25\n{e}") from e

    bm25 = BM25Okapi([tokenize(d) for d in corpus])
    raw_scores = bm25.get_scores(tokenize(query))

    results = [
        {**meta[i], "score": float(s)}
        for i, s in enumerate(raw_scores)
        if s >= min_score
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rank_with_tfidf(
    query: str,
    corpus: list[str],
    meta: list[dict[str, str]],
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Rank corpus documents against a query using TF-IDF + cosine similarity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError(f"scikit-learn not installed — run: pip install scikit-learn\n{e}") from e

    vec = TfidfVectorizer()
    mat = vec.fit_transform(corpus)
    q_vec = vec.transform([query])
    scores = cosine_similarity(q_vec, mat).flatten()

    results = [
        {**meta[i], "score": float(s)}
        for i, s in enumerate(scores)
        if s >= min_score
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rank_with_embeddings(
    query: str,
    corpus: list[str],
    meta: list[dict[str, str]],
    top_k: int,
    min_score: float,
    model_name: str,
) -> list[dict[str, Any]]:
    """Rank corpus documents against a query using dense embeddings + cosine similarity."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as e:
        raise ImportError(
            f"sentence-transformers not installed — run: pip install sentence-transformers\n{e}"
        ) from e

    log.info(f"## Loading embedding model **`{model_name}`** ...")
    model = SentenceTransformer(model_name)
    corpus_emb = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)

    scores = util.cos_sim(query_emb, corpus_emb).squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]

    results = [
        {**meta[i], "score": float(s)}
        for i, s in enumerate(scores)
        if s >= min_score
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank(
    query: str,
    results: list[dict[str, Any]],
    rerank_model: str,
) -> list[dict[str, Any]]:
    """Rerank candidate results using a cross-encoder model."""
    if not results:
        return results
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            f"sentence-transformers not installed — run: pip install sentence-transformers\n{e}"
        ) from e

    log.info(f"## Loading cross-encoder reranker **`{rerank_model}`** ...")
    ce = CrossEncoder(rerank_model)
    pairs = [(query, r["doc"]) for r in results]
    rerank_scores = ce.predict(pairs)

    for r, s in zip(results, rerank_scores):
        r["rerank_score"] = float(s)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


def deduplicated(items: list[str]) -> list[str]:
    """Return a list of unique items preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def determine_best_tools(
    prompt: str,
    schema: str,
    model: str = "bm25",
    top_k: int = 5,
    min_score: float = 0.0,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    log_level: str = "INFO",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Determine the best tool matches for a given prompt.

    Args:
        prompt: Search query (e.g. 'get date')
        schema: Path to JSON schema file with 'uses' dict
        model: Retrieval model: bm25 | tfidf | <sentence-transformer-model-name>
        top_k: Number of top results
        min_score: Minimum retrieval score threshold
        rerank: Whether to apply cross-encoder reranker after retrieval
        rerank_model: Cross-encoder model for reranking
        log_level: Log level

    Returns:
        dict with keys: query, model, reranked, best_files, best_uses, results
    """
    if verbose:
        log.info(f"## Loading schema from `{schema}`")
    uses = load_schema(schema)
    if verbose:
        log.info(f"## Loaded **{len(uses)}** file(s) from schema")

    if verbose:
        log.info(f"## Building corpus")
    corpus, meta = build_corpus(uses)
    # print(corpus)
    if verbose:
        log.info(f"## Corpus: **{len(corpus)}** documents across {len(uses)} file(s)")

    model_key = model.lower()
    if verbose:
        log.info(f"## Retrieving with model **`{model}`**, query: `{prompt}`")

    if model_key == "bm25":
        results = rank_with_bm25(prompt, corpus, meta, top_k, min_score)
    elif model_key == "tfidf":
        results = rank_with_tfidf(prompt, corpus, meta, top_k, min_score)
    else:
        results = rank_with_embeddings(
            prompt, corpus, meta, top_k, min_score, model
        )

    if verbose:
        log.info(f"## Retrieved **{len(results)}** candidate(s) above min_score={min_score}")

    if rerank:
        if not results:
            log.info(f"## Skipping reranker — no candidates to rerank")
        else:
            results = rerank(prompt, results, rerank_model)
            if verbose:
                log.info(f"## Reranking complete")

    best_files = deduplicated(r["file"] for r in results)
    best_uses = deduplicated(f"{r['file']}:{r['func']}" for r in results)

    if verbose:
        log.info(f"### Best files ({len(best_files)}):")
        for i, f in enumerate(best_files, 1):
            log.info(f"  {i}. `{f}`")

        log.info(f"### Coder Private Repo Best Tools ({len(best_uses)}):")
        for i, u in enumerate(best_uses, 1):
            score_key = "rerank_score" if "rerank_score" in results[i - 1] else "score"
            score = results[i - 1].get(score_key, 0.0)
            log.info(f"  {i}. `{u}`  score={score:.4f}")

    output = {
        "query": prompt,
        "model": model,
        "reranked": rerank,
        "best_files": best_files,
        "best_uses": best_uses,
        "results": [
            {
                "file": r["file"],
                "func": r["func"],
                "description": r["description"],
                "score": r.get("rerank_score", r["score"]),
                "retrieval_score": r["score"],
            }
            for r in results
        ],
    }
    return output


def main() -> None:
    """CLI entry point: rank tool matches and print JSON output."""
    tool_uses_index_file = os.getenv('CODER_TOOL_USES_INDEX', './.ai/AGENT.repo_uses.python.tools.json')
    parser = argparse.ArgumentParser(description="Rank tool matches from a uses schema with BM25 / TF-IDF / embeddings + optional reranker")
    parser.add_argument("-p", "--prompt", required=True, help="Search query (e.g. 'get date')")
    parser.add_argument("-s", "--schema", default=tool_uses_index_file, help="Path to JSON schema file with 'uses' dict (default: stdin)")
    parser.add_argument("-m", "--model", default="bm25", help="Retrieval model: bm25 | tfidf | <sentence-transformer-model-name>  (default: bm25)")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of top results (default: 5)")
    parser.add_argument("-w", "--min-score", type=float, default=0.0, help="Minimum retrieval score threshold (default: 0.0)")
    parser.add_argument("-r", "--rerank", action="store_true", help="Apply cross-encoder reranker after retrieval (improves quality, slower)")
    parser.add_argument("-R", "--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model for reranking (default: ms-marco-MiniLM-L-6-v2)")
    parser.add_argument("-l", "--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args()

    try:
        result = determine_best_tools(
            prompt=args.prompt,
            schema=args.schema,
            model=args.model,
            top_k=args.top_k,
            min_score=args.min_score,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
            log_level=args.log_level,
        )
        print(json.dumps(result, indent=2))

    except Exception:
        log.error(f"## Fatal error\n```\n{traceback.format_exc()}\n```")
        sys.exit(1)


if __name__ == "__main__":
    main()
