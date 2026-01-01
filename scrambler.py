#!/usr/bin/env python3
"""
Scramble/descramble dataset files using XOR encryption.

Usage:
    python scrambler.py -s dataset.json    # Scramble to dataset.scr
    python scrambler.py -d dataset.scr     # Descramble to dataset.json
    python scrambler.py -d dataset.scr -o output.json  # Custom output path
"""

import argparse
import sys
from pathlib import Path

KEY = b"MisguidedAttention2025"


def xor_bytes(data: bytes, key: bytes = KEY) -> bytes:
    """XOR data with key (works for both scramble and descramble)."""
    return bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))


def scramble(input_path: str, output_path: str | None = None) -> str:
    """Scramble a JSON file to .scr format."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out = Path(output_path) if output_path else path.with_suffix(".scr")

    with open(path, "rb") as f:
        data = f.read()

    with open(out, "wb") as f:
        f.write(xor_bytes(data))

    return str(out)


def descramble(input_path: str, output_path: str | None = None) -> str:
    """Descramble a .scr file to JSON format."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out = Path(output_path) if output_path else path.with_suffix(".json")

    with open(path, "rb") as f:
        data = f.read()

    with open(out, "wb") as f:
        f.write(xor_bytes(data))

    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Scramble/descramble dataset files")
    parser.add_argument("mode", choices=["-s", "-d"], help="-s to scramble, -d to descramble")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path (optional)")

    args = parser.parse_args()

    try:
        if args.mode == "-s":
            out = scramble(args.input, args.output)
            print(f"Scrambled to {out}")
        else:
            out = descramble(args.input, args.output)
            print(f"Descrambled to {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
