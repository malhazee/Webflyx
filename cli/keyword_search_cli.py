import argparse
import math
from lib.keyword_search import (
    InvertedIndex,
    tokenize_text,
    tokinize_term,
    BM25_K1,
    BM25_B,
    bm25_tf_command,
    bm25_idf_command,
    build_command,
    load_movies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and save inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords"
    )
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate Inverse Document Frequency fro a term"
    )
    idf_parser.add_argument("term", help="Term to calculate IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate TF-IDF score for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF socre for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first.")
                return

            query_tokens = tokenize_text(args.query)

            seen_ids = set()
            result = []

            for token in query_tokens:
                doc_ids = idx.get_documents(token)
                for doc_id in doc_ids:
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        result.append(idx.docmap[doc_id])
                        if len(result) == 5:
                            break
                if len(result) == 5:
                    break

            for i, doc in enumerate(result, start=1):
                print(f"{i}. ({doc['id']}) {doc['title']}")

        case "build":
            build_command()

        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first.")
                return
            token = tokinize_term(args.term)
            print(idx.get_tf(args.doc_id, token))

        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first")
                return

            token = tokinize_term(args.term)
            total_doc_count = len(idx.docmap)
            term_match_doc_count = len(idx.get_documents(token))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first.")
                return
            token = tokinize_term(args.term)

            tf = idx.get_tf(args.doc_id, token)

            total_doc_count = len(idx.docmap)
            term_match_doc_count = len(idx.get_documents(token))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            tf_idf = tf * idf

            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first.")
                return

            result = idx.bm25_search(args.query, limit=5)
            for i, (doc, score) in enumerate(result, start=1):
                print(f"{i}. ({doc['id']}) {doc['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()