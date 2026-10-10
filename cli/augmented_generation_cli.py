import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")


    summarize_parser = subparsers.add_parser("summarize", help="Summarize multiple search results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize (default: 5)")

        # 3. أمر citations
    citations_parser = subparsers.add_parser("citations", help="Answer query with citations from search results")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve (default: 5)")

    # 4. أمر question
    question_parser = subparsers.add_parser(
        "question", help="Answer a question based on search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to retrieve (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            movies = load_movies()
            search = HybridSearch(movies)
            results = search.rrf_search(query, k=60, limit=5)

            docs_list = []
            for doc in results:
                desc = doc.get("description", "") or doc.get("document", "")
                if isinstance(desc, dict):
                    desc = desc.get("description", "")
                docs_list.append(f"{doc['title']}: {desc}")
            docs = "\n\n".join(docs_list)

            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service. Your task is to provide a natural-language answer to the user's query based on documents retrieved during search. Provide a comprehensive answer that addresses the user's query.

Query: {query}
Documents:
{docs}

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            rag_response = response.choices[0].message.content.strip()

            print("Search Results:")
            for doc in results:
                print(f"- {doc['title']}")
            print("RAG Response:")
            print(rag_response)

        case "summarize":
            query = args.query

            # 1. جلب الأفلام باستخدام RRF بالـ limit المحدد
            movies = load_movies()
            search = HybridSearch(movies)
            results = search.rrf_search(query, k=60, limit=args.limit)

            # 2. تجهيز الوثائق للـ Prompt
            docs_list = []
            for doc in results:
                desc = doc.get("description", "") or doc.get("document", "")
                if isinstance(desc, dict):
                    desc = desc.get("description", "")
                docs_list.append(f"{doc['title']}: {desc}")
            results_str = "\n\n".join(docs_list)

            # 3. إعداد الاتصال بـ OpenRouter
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            # 4. Prompt التلخيص المطلوب في الدرس
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail. The goal is to provide comprehensive information so that users know what their options are.

Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie. This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}
Search results:
{results_str}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content.strip()

            # 5. طباعة النتائج والملخص بالتنسيق المطلوب
            print("Search Results:")
            for doc in results:
                print(f"- {doc['title']}")
            print("LLM Summary:")
            print(summary)

        case "citations":
            query = args.query

            # 1. جلب الأفلام باستخدام RRF
            movies = load_movies()
            search = HybridSearch(movies)
            results = search.rrf_search(query, k=60, limit=args.limit)

            # 2. تجهيز الوثائق مرقمة بـ [1]، [2]، ... لتسهيل الاستشهاد بها
            docs_list = []
            for i, doc in enumerate(results, start=1):
                desc = doc.get("description", "") or doc.get("document", "")
                if isinstance(desc, dict):
                    desc = desc.get("description", "")
                docs_list.append(f"[{i}] {doc['title']}: {desc}")
            documents = "\n\n".join(docs_list)

            # 3. إعداد الاتصال بـ OpenRouter
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            # 4. Prompt الاستشهاد بالمصادر المحدد في الدرس
            prompt = f"""Answer the query below and give information based on the provided documents. The answer should be tailored to users of Webflyx, a movie streaming service. If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}
Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()

            # 5. طباعة النتائج والرد المرفق بالاقتباسات بالتنسيق المطلوب
            print("Search Results:")
            for doc in results:
                print(f"- {doc['title']}")
            print("LLM Answer:")
            print(answer)

        case "question":
            question = args.question

            # 1. جلب الأفلام باستخدام RRF
            movies = load_movies()
            search = HybridSearch(movies)
            results = search.rrf_search(question, k=60, limit=args.limit)

            # 2. تجهيز الوثائق لتمثل السياق (Context)
            docs_list = []
            for doc in results:
                desc = doc.get("description", "") or doc.get("document", "")
                if isinstance(desc, dict):
                    desc = desc.get("description", "")
                docs_list.append(f"{doc['title']}: {desc}")
            context = "\n\n".join(docs_list)

            # 3. إعداد الاتصال بـ OpenRouter
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            # 4. الـ Prompt الموجه للإجابة المباشرة
            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}
Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()

            # 5. طباعة النتائج والإجابة بالتنسيق المطلوب
            print("Search Results:")
            for doc in results:
                print(f"- {doc['title']}")
            print("Answer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()