from pathlib import Path
import os

"""
Central configuration for the CCUS analysis pipeline.

Edit `API_BASE_URL` here (or set the CCUS_API_BASE_URL environment variable)
to change which OpenParliament API server the pipeline talks to.

Edit `OUTPUT_DIR` to move the step output files.
"""

# Base URL for the OpenParliament API used by all steps.
API_BASE_URL: str = os.environ.get("CCUS_API_BASE_URL", "http://api.openparliament.ca")

# Default root directory for pipeline outputs:
#   parliament/ccus_analysis/output/
OUTPUT_ROOT: Path = Path(__file__).resolve().parent / "output"

# Subdirectories for different artifact types
JSON_DIR: Path = OUTPUT_ROOT / "json"
CSV_DIR: Path = OUTPUT_ROOT / "csv"
VIS_DIR: Path = OUTPUT_ROOT / "vis"

# Backward-compatible alias used by step5_vis.py
OUTPUT_DIR: Path = OUTPUT_ROOT

# Ensure directories exist when the module is imported
for _d in (OUTPUT_ROOT, JSON_DIR, CSV_DIR, VIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

