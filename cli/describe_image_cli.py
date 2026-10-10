import argparse
import base64
import mimetypes
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Multimodal query rewriting")
    parser.add_argument("--image", required=True, type=str, help="Path to image file")
    parser.add_argument(
        "--query", required=True, type=str, help="Text query to rewrite"
    )
    args = parser.parse_args()

    # 1. تحديد نوع الصورة وقراءتها كبيانات ثنائية
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    # 2. تشفير الصورة إلى base64 data URL
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    # 3. إعداد عميل OpenRouter
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # 4. بناء الـ Prompt والرسالة متعددة الوسائط
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    # 5. إرسال الطلب إلى نموذج مجاني يدعم الوسائط المتعددة
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()