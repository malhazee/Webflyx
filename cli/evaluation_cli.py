import argparse
import json
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of result to evaluate (k precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    movies = load_movies()
    search = HybridSearch(movies)

    print(f"k={limit}")

    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [doc["title"] for doc in results]

        relevant_retrieved = [
            title for title in retrieved_titles if title in relevant_docs
        ]
        precision = (
            len(relevant_retrieved) / len(retrieved_titles)
            if retrieved_titles
            else 0.0
        )
        recall = (
            len(relevant_retrieved) / len(relevant_docs)
            if relevant_docs
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")

if __name__ == "__main__":
    main()