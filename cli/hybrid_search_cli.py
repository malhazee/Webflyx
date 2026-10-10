import argparse
import os
import time
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from lib.hybrid_search import min_max_normalize, HybridSearch
from lib.keyword_search import load_movies
from sentence_transformers import CrossEncoder

def correct_spelling(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY enviroment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content.strip()

def rewrite_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY enviroment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable. Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic
Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
If you cannot improve the query, output the original unchanged. Output only the rewritten query text, nothing else.
User query: "{query}" """
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content.strip()

def rerank_individual(query: str, doc: dict) -> float:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    desc = doc.get("description", "") or doc.get("document", "")
    if isinstance(desc, dict):
        desc = desc.get("description", "")

    prompt = f"""Rate how well this movie matches the search query.
Query: "{query}"
Movie: {doc.get("title", "")} - {desc}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match). Output ONLY the number in your response, no other text or explanation.

Score:"""

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"(\d+(\.\d+)?)", content)
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0

def rerank_batch(query: str, docs: list[dict]) -> list[int]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    doc_list_lines = []
    for doc in docs:
        doc_id = doc.get("id") or doc.get("document", {}).get("id")
        desc = doc.get("description", "") or doc.get("document", "")
        if isinstance(desc, dict):
            desc = desc.get("description", "")
        doc_list_lines.append(f"{doc_id}: {doc['title']} - {desc[:150]}")

    doc_list_str = "\n".join(doc_list_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.
Query: "{query}"
Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first. Your response must be a raw JSON array of integers. Do not wrap the JSON in Markdown. Do not use a ```json code block. Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()
    if "```" in content:
        content = re.sub(r"```(json)?", "", content).replace("```", "").strip()

    try:
        ranked_ids = json.loads(content)
        return ranked_ids
    except Exception:
        return [doc.get("id") or doc.get("document", {}).get("id") for doc in docs]

def expand_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Expand the user-provided movie search query below with related terms. Add synonyms and related concepts that might appear in movie descriptions. Keep expansions relevant and focused. Output only the additional terms; they will be appended to the original query. Examples: - "scary bear movie" -> "scary horror grizzly bear movie terrifying film" - "action movie with bear" -> "action thriller bear chase fight adventure" - "comedy with bear" -> "comedy funny bear humor lighthearted" User query: "{query}" """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    expanded_terms = response.choices[0].message.content.strip()
    return f"{query} {expanded_terms}"

def evaluate_results(query: str, results: list[dict]) -> list[int]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    formatted_results = []
    for doc in results:
        desc = doc.get("description", "") or doc.get("document", "")
        if isinstance(desc, dict):
            desc = desc.get("description", "")
        formatted_results.append(f"{doc['title']} - {desc[:150]}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
Query: "{query}"
Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.
Return ONLY the scores in the same order you were given the documents.
Return a valid JSON list, nothing else.

For example:
[2, 0, 3, 2, 0, 1]"""

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r"\[[\d\s,]+\]", content)
            if match:
                parsed = json.loads(match.group(0))
                if len(parsed) == len(results):
                    return parsed
            parsed = json.loads(content)
            if len(parsed) == len(results):
                return parsed
        except Exception:
            pass
        time.sleep(2)

    return [0] * len(results)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to nromalize")

    weighted_parser = subparser.add_parser("weighted-search", help=" Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help=" Weight between keyword and semanic search (default: 0.5)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of result to return (default: 5)")

    rrf_parser = subparser.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", "--k", type=int, default=60, help="RRF k pararmeter (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of result to return (default: 5)")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Re-ranking method")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Evaluate search results using an LLM")

    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normliazed = min_max_normalize(args.scores)
            for score in normliazed:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            search = HybridSearch(movies)
            results = search.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"  MB25: {res['bm25_score']:.3f}, semantic: {res['semantic_score']:.3f}")
                print(f"  {res['description'][:100]}...\n")

        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                enhanced_query = correct_spelling(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query  = enhanced_query
            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
            elif args.enhance == "expand":
                enhanced_query = rewrite_query(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query

            movies = load_movies()
            search = HybridSearch(movies)
            fetch_limit = args.limit * 5 if args.rerank_method else args.limit
            results = search.rrf_search(query, args.k, fetch_limit)

            if args.rerank_method == "individual":
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...\n")
                for doc in results:
                    score = rerank_individual(query, doc)
                    doc["rerank_score"] = score
                    time.sleep(3)
                results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                results = results[:args.limit]
            elif args.rerank_method == "batch":
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...\n")
                ranked_ids = rerank_batch(query, results)
                for doc in results:
                    doc_id = doc.get("id") or doc.get("document", {}).get("id")
                    if doc_id in ranked_ids:
                        doc["rerank_rank"] = ranked_ids.index(doc_id) + 1
                    else:
                        doc["rerank_rank"] = 999
                results.sort(key=lambda x: x.get("rerank_rank", 999))
                results = results[:args.limit]
            elif args.rerank_method == "cross_encoder":
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...\n")
                pairs = []
                for doc in results:
                    desc = doc.get("description", "") or doc.get("document", "")
                    if isinstance(desc, dict):
                        desc = desc.get("description", "")
                    pairs.append([query, f"{doc.get('title', '')} - {desc}"])

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)

                for doc, score in zip(results, scores):
                    doc["cross_encoder_score"] = float(score)
                results.sort(key=lambda x: x.get("cross_encoder_score", float("-inf")), reverse=True)
                results = results[:args.limit]

            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
                if "cross_encoder_score" in res:
                    print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")
                elif "rerank_score" in res:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                elif "rerank_rank" in res:
                    print(f"   Re-rank Rank: {res['rerank_rank']}")
                print(f"   RRF Score: {res['rrf_score']:.3f}")
                print(f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
                print(f"   {res['description'][:100]}...\n")

            if args.evaluate:
                scores = evaluate_results(query, results)
                for i, (res, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {res['title']}: {score}/3")
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()