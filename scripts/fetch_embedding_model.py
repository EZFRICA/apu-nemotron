#!/usr/bin/env python3
"""
Populate the local ONNX embedding model cache. Ported from Akili.

RUN THIS ONCE, ON A CONNECTED MACHINE, BEFORE DEPLOYMENT.

The local embedder never downloads at runtime: on a machine with no connectivity
a download attempt hangs at the first question instead of failing. So the model
has to be in the cache directory before the app is used, and the cache directory
ships with the deployment.

    uv run python scripts/fetch_embedding_model.py
    uv run python scripts/fetch_embedding_model.py --cache-dir /media/usb/models

Then copy the cache directory onto each target device (or image it in).
Roughly 240MB for the default model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apu import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.LOCAL_EMBEDDING_MODEL)
    parser.add_argument("--cache-dir", default=config.LOCAL_EMBEDDING_CACHE_DIR)
    parser.add_argument(
        "--expected-dim", type=int, default=config.LOCAL_EMBEDDING_DIM,
        help="fail if the model does not produce this many dimensions",
    )
    args = parser.parse_args()

    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("fastembed is not installed. Run `uv sync` first.",
              file=sys.stderr)
        return 1

    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if args.model not in supported:
        print(f"'{args.model}' is not available in this fastembed version.\n",
              file=sys.stderr)
        print("Models offering the same dimension:", file=sys.stderr)
        for m in TextEmbedding.list_supported_models():
            if m.get("dim") == args.expected_dim:
                print(f"  {m['model']}  ({m.get('size_in_GB')} GB)", file=sys.stderr)
        return 1

    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"model:     {args.model}")
    print(f"cache dir: {args.cache_dir}")
    print("\nDownloading (this needs a network connection)...")

    t0 = time.time()
    model = TextEmbedding(model_name=args.model, cache_dir=args.cache_dir)
    vector = next(iter(model.embed(["vérification du modèle"])))
    elapsed = time.time() - t0

    dim = len(vector)
    print(f"\nfetched in {elapsed:.1f}s — dimension {dim}")

    if dim != args.expected_dim:
        print(
            f"\nDIMENSION MISMATCH: model produced {dim}, config expects "
            f"{args.expected_dim}.\nSet LOCAL_EMBEDDING_DIM={dim} or choose a "
            f"different model.",
            file=sys.stderr,
        )
        return 1

    # Prove the cache-only path the runtime actually uses will work on the target.
    TextEmbedding(model_name=args.model, cache_dir=args.cache_dir,
                  local_files_only=True)
    print("verified: the model loads from cache with no network.")
    print(f"\nNow copy '{args.cache_dir}' onto the target device(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
