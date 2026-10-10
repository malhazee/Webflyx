import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # أمر verify_image_embedding
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image file")

    # أمر image_search الجديد
    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image"
    )
    image_search_parser.add_argument("image_path", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']} (similarity: {res['similarity']:.3f})")
                print(f"{res['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()