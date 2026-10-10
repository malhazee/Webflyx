import os
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[dict]:
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        bm25_score = [score for _, score in bm25_result]
        semantic_score = [doc["score"] for doc in semantic_result]

        normalized_bm25 = min_max_normalize(bm25_score)
        normalized_semantic = min_max_normalize(semantic_score)

        combined = {}
        for (doc, _), score in zip(bm25_result, normalized_bm25):
            doc_id = doc["id"]
            combined[doc_id] = {
                "document": doc,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "bm25_score": score,
                "semantic_score": 0.0,
            }

        for doc, score in zip(semantic_result, normalized_semantic):
            doc_id = doc["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = score
            else:
                combined[doc_id] = {
                    "document": doc,
                    "title": doc["title"],
                    "description": doc.get("description", doc.get("document", "")),
                    "bm25_score": 0.0,
                    "semantic_score": score,
                }
        results = []
        for doc_id, data in combined.items():
            h_score = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
            data["hybrid_score"] = h_score
            results.append(data)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]   
                 

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        combined = {}
        for rank, (doc, _) in enumerate(bm25_result, start=1):
            doc_id = doc["id"]
            combined[doc_id] = {
                "id": doc_id,
                "document": doc,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "bm25_rank":rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, (doc) in enumerate(semantic_result, start=1):
                    doc_id = doc["id"]
                    score = rrf_score(rank, k)
                    if doc_id in combined:
                        combined[doc_id]["semantic_rank"] = rank
                        combined[doc_id]["rrf_score"] += score
                    else:
                        combined[doc_id] = {
                            "id": doc_id,
                            "document": doc,
                            "title": doc["title"],
                            "description": doc.get("description", doc.get("document", "")),
                            "bm25_rank": None,
                            "semantic_rank": rank,
                            "rrf_score": rrf_score(rank, k),
                        }
        results = list(combined.values())
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:limit]
