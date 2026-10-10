import json
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter
import math

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies():
    with open("data/movies.json", mode="r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def tokinize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token, got {len(tokens)}")
    return tokens[0]


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    remove_punctuation = str.maketrans("", "", string.punctuation)

    with open("data/stopwords.txt", mode="r", encoding="utf-8") as f:
        stopwords = {
            w.lower().translate(remove_punctuation) for w in f.read().splitlines()
        }
    clean_text = text.lower().translate(remove_punctuation)
    tokens = clean_text.split()
    return [stemmer.stem(t) for t in tokens if t not in stopwords]


class InvertedIndex:
    def __init__(self):
        self.index_path = "cache/index.pkl"
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        if term not in self.index:
            return []
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            full_text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, full_text)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not (
            os.path.exists("cache/index.pkl")
            and os.path.exists("cache/docmap.pkl")
            and os.path.exists("cache/term_frequencies.pkl")
            and os.path.exists("cache/doc_lengths.pkl")
        ):
            raise FileNotFoundError("Index files not found")

        with open("cache/index.pkl", mode="rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", mode="rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", mode="rb") as f:
            self.term_frequencies = pickle.load(f)

        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1.0

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5):
        query_tokens = tokenize_text(query)
        scores = {}

        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in sorted_docs[:limit]]


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0

    token = tokinize_term(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0

    token = tokinize_term(term)
    return idx.get_bm25_idf(token)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
