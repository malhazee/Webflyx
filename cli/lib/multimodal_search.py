import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_movies():
    with open("data/movies.json", mode="r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


class MultimodalSearch:
    def __init__(self, documents: list[dict] = None, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        
        # دمج العنوان والوصف لكل فيلم
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        
        # تشفير جميع النصوص إذا وجدت وثائق
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = []

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        results = []

        # حساب تشابه جيب التمام بين الصورة وكل نصوص الأفلام
        for i, doc in enumerate(self.documents):
            sim = cosine_similarity(image_embedding, self.text_embeddings[i])
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "description": doc.get("description", ""),
                "similarity": sim,
            })

        # الترتيب تنازلياً وإرجاع أول 5 أفلام
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    movies = load_movies()
    search = MultimodalSearch(movies)
    return search.search_with_image(image_path)