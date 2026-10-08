"""Candidate fusion algorithms: Weighted Score Fusion and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to the range [0.0, 1.0]."""
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    if s_max - s_min < 1e-9:
        return [1.0] * len(scores)
    return [(s - s_min) / (s_max - s_min) for s in scores]


def weighted_score_fusion(
    semantic_results: list[dict],
    lexical_results: list[dict],
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    top_k: int = 10,
) -> list[dict]:
    """Fuse candidates using min-max normalized weighted linear combination."""
    fused: dict[str, dict] = {}

    # Extract raw scores for normalization
    sem_scores = [r.get("semantic_score", r.get("score", 0.0)) for r in semantic_results]
    lex_scores = [r.get("lexical_score", r.get("score", 0.0)) for r in lexical_results]

    norm_sem = normalize_scores(sem_scores)
    norm_lex = normalize_scores(lex_scores)

    # Process semantic candidates
    for i, res in enumerate(semantic_results):
        cid = res["chunk_id"]
        item = dict(res)
        s_norm = norm_sem[i] if i < len(norm_sem) else 0.0
        item["norm_semantic_score"] = s_norm
        item["semantic_score"] = sem_scores[i]
        item["lexical_score"] = 0.0
        item["norm_lexical_score"] = 0.0
        item["fusion_score"] = semantic_weight * s_norm
        fused[cid] = item

    # Process lexical candidates
    for i, res in enumerate(lexical_results):
        cid = res["chunk_id"]
        l_norm = norm_lex[i] if i < len(norm_lex) else 0.0
        l_raw = lex_scores[i] if i < len(lex_scores) else 0.0

        if cid in fused:
            fused[cid]["lexical_score"] = l_raw
            fused[cid]["norm_lexical_score"] = l_norm
            fused[cid]["fusion_score"] += lexical_weight * l_norm
        else:
            item = dict(res)
            item["semantic_score"] = 0.0
            item["norm_semantic_score"] = 0.0
            item["lexical_score"] = l_raw
            item["norm_lexical_score"] = l_norm
            item["fusion_score"] = lexical_weight * l_norm
            fused[cid] = item

    # Sort descending by fusion_score
    ranked = sorted(fused.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)

    # Set final score to fusion_score for downstream consumption
    for r in ranked:
        r["score"] = r["fusion_score"]

    return ranked[:top_k]


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    lexical_results: list[dict],
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Fuse candidates using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Accumulate RRF scores from semantic ranking
    for rank, res in enumerate(semantic_results):
        cid = res["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
        if cid not in doc_map:
            doc_map[cid] = dict(res)
            doc_map[cid]["semantic_score"] = res.get("semantic_score", res.get("score", 0.0))
            doc_map[cid]["lexical_score"] = 0.0

    # Accumulate RRF scores from lexical ranking
    for rank, res in enumerate(lexical_results):
        cid = res["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
        if cid not in doc_map:
            doc_map[cid] = dict(res)
            doc_map[cid]["semantic_score"] = 0.0
        doc_map[cid]["lexical_score"] = res.get("lexical_score", res.get("score", 0.0))

    ranked = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True):
        item = doc_map[cid]
        item["fusion_score"] = rrf_score
        item["score"] = rrf_score
        ranked.append(item)

    return ranked[:top_k]
