import argparse
import json
import re
from lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
    semantic_chunk_text,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
)

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        return json.load(f)["movies"]
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers  = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the embedding model")
    subparsers.add_parser("verify_embeddings", help="Verify or build movie embeddings")
    subparsers.add_parser("embed_chunks", help="Generate or load chunked embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for text string")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an enbedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of orverlaping words")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text based on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences")

    search_chunk_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic embeddings")
    search_chunk_parser.add_argument("query", type=str, help="Search query")
    search_chunk_parser.add_argument("--limit", type=int, default=5, help="Number of result to return")

    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)    

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search = SemanticSearch()
            with open("data/movies.json", "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            search.load_or_create_embeddings(documents)

            results = search.search(args.query, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['description']}")

        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            words = args.text.split()
            chunks = []
            step = args.chunk_size - args.overlap
            i = 0

            while i < len(words):
                chunk = words[i : i + args.chunk_size]
                chunks.append(" ".join(chunk))
                if i + args.chunk_size >= len(words):
                    break
                i += step

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            with open("data/movies.json", "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            documents = load_movies()
            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(documents)

            results = search.search_chunks(args.query, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"  {res['document']}...")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()