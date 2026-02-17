"""
Text Generation Script

Generate Swahili text from a trained model.

Usage:
    # Single prompt
    python scripts/generate.py \
        --checkpoint checkpoints/latest.pt \
        --prompt "Tanzania ni nchi" \
        --max_tokens 200

    # Interactive mode
    python scripts/generate.py \
        --checkpoint checkpoints/latest.pt \
        --interactive
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Swahili text")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/swahili_bpe.json")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--strategy", type=str, default="top_p", choices=["greedy", "top_k", "top_p"])
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    args = parser.parse_args()

    # TODO: Implement generation
    # 1. Load model and tokenizer
    # 2. Create Generator instance
    # 3. If interactive: loop reading prompts from stdin
    # 4. If prompt: generate and print

    if args.interactive:
        print("Swahili-100 Interactive Generation")
        print("Type a prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if prompt.strip().lower() == "quit":
                    break
                # TODO: Generate and print
                print("[Generation not yet implemented]\n")
            except (EOFError, KeyboardInterrupt):
                break
    else:
        if args.prompt is None:
            parser.error("Provide --prompt or use --interactive")
        # TODO: Generate from prompt
        print(f"Prompt: {args.prompt}")
        print("[Generation not yet implemented]")


if __name__ == "__main__":
    main()
