from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
import re


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
            if not text or not text.strip():
                raise ValueError("Input text connot be empty or whitespace only")
    
            embeddings = self.model.encode([text])
            return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        cache_path = "cache/movie_embeddings.npy"
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")

        query_embedding = self.generate_embedding(query)
        result = []
        for i, doc in enumerate(self.documents):
            sim = cosine_similarity(query_embedding, self.embeddings[i])
            result.append((sim, doc))

        result.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in result[:limit]
        ]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(self.documents):
            desc = doc.get("description", "")
            if not desc or not desc.strip():
                continue

            doc_chunks = semantic_chunk_text(desc, max_chunk_size=4, overlap=1)
            total_chunks = len(doc_chunks)

            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        cache_embeddings_path = "cache/chunk_embeddings.npy"
        cache_metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(cache_embeddings_path) and os.path.exists(cache_metadata_path):
            self.chunk_embeddings = np.load(cache_embeddings_path)
            with open(cache_metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        # 1. توليد متجه الاستعلام
        query_embedding = self.generate_embedding(query)
        chunk_scores = []

        # 2. حساب التشابه مع كل قطعة
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            sim = cosine_similarity(query_embedding, chunk_embedding)
            meta = self.chunk_metadata[i]
            chunk_scores.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": sim,
            })

        # 3. تجميع الدرجات لكل فيلم (الاحتفاظ بأعلى درجة لكل فيلم)
        movie_scores = {}
        for cs in chunk_scores:
            m_idx = cs["movie_idx"]
            s = cs["score"]
            if m_idx not in movie_scores or s > movie_scores[m_idx]:
                movie_scores[m_idx] = s

        # 4. الترتيب تنازلياً
        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 5. بناء النتائج (قص الوصف إلى 100 حرف)
        results = []
        for movie_idx, score in sorted_movie_scores[:limit]:
            doc = self.documents[movie_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 3),
            })

        return results
      
def semantic_chunk_text(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    text = text.strip()

    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    if len (raw_sentences) == 1 and not raw_sentences[0].endswith((".", "!", "?")):
        sentences = [raw_sentences[0].strip()]
    else:
        sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    step = max_chunk_size - overlap
    i = 0

    while i < len(sentences):
        chunk_sentences =[s.strip() for s in sentences[i : i + max_chunk_size] if s.strip()]
        chunk_str = " ".join(chunk_sentences).strip()

        if chunk_str:
            chunks.append(chunk_str)

        if i + max_chunk_size >= len(sentences):
            break
        i += step

    return chunks

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    search = SemanticSearch()
    with open("data/movies.json", "r", encoding="utf-8") as f:
        documents = json.load(f)["movies"]

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))