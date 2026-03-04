#!/usr/bin/env python3
"""Entry point for running TimeQA baseline experiments."""

import sys
from pathlib import Path

# Add src to Python path so we can import the package
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main function from the package
from timeqa_baseline_lab.run import main

if __name__ == "__main__":
    main()
