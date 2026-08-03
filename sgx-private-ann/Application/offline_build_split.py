"""Thin entrypoint for offline HNSW build + split."""

#!/usr/bin/env python3
from __future__ import annotations

import sys

from hnsw_split_faiss import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
