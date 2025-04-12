from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
TRAINED_MODELS_DIR = OUTPUTS_DIR / "trained_models"
LOGS_DIR = OUTPUTS_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
