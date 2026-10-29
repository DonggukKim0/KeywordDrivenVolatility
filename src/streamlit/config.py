"""Central configuration for paths to existing project artifacts.

Keep data, image, and model artifacts in their current locations. Set the
corresponding environment variables when those locations are ready to connect.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Existing project artifacts stay in their current locations. Override these
# paths with environment variables when running the dashboard elsewhere.
DATA_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", PROJECT_ROOT / "data"))
EDA_OUTPUT_DIR = Path(
    os.environ.get("DASHBOARD_FIGURES_DIR", PROJECT_ROOT / "eda-outputs")
)
MODEL_OUTPUTS_DIR = Path(
    os.environ.get("DASHBOARD_MODEL_OUTPUTS_DIR", PROJECT_ROOT / "model-outputs")
)
EDA_STEP01_DIR = EDA_OUTPUT_DIR / "step01"
EDA_STEP02_DIR = EDA_OUTPUT_DIR / "step02"
EDA_STEP03_DIR = EDA_OUTPUT_DIR / "step03"
EDA_STEP01_TABLES_DIR = EDA_STEP01_DIR / "tables"
EDA_STEP02_TABLES_DIR = EDA_STEP02_DIR / "tables"
EDA_STEP03_TABLES_DIR = EDA_STEP03_DIR / "tables"
MODEL_RESULTS_DIR = MODEL_OUTPUTS_DIR / "results"
