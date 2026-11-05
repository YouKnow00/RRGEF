"""RRGEF: relative reliability-guided evidential fusion for multi-omics survival prediction."""
from rrgef.rrgef import RRGEF
from rrgef.preprocessing import select_fit_apply
from rrgef.metrics import concordance_index

__version__ = "1.0.0"
__all__ = ["RRGEF", "select_fit_apply", "concordance_index"]
