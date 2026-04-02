from .loader import load_data
from .synthetic import generate_village_data
from .preprocessing import preprocess, compute_composite_indicators

__all__ = ["load_data", "generate_village_data", "preprocess", "compute_composite_indicators"]
