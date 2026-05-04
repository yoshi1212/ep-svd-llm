#!/usr/bin/env python3
"""CLI for post-compression fine-tuning of compressed causal LMs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this file as a thin public entrypoint; implementation lives in the
# package so runners and tests can reuse it without shelling out.
from ep_svd_llm.finetune.post_compression import main


if __name__ == "__main__":
    main()
