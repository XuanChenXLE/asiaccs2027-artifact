"""Thin entrypoint for streaming offline HNSW build + split."""

#!/usr/bin/env python3
from __future__ import annotations

import sys

from hnsw_split_faiss_streaming import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
