
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .MDP import MDP
    from .LMDP import LMDP
    from .LMDP_TDR import LMDP_TDR

from scipy.sparse import csr_matrix
import numpy as np

def compare_models(src: MDP | LMDP | LMDP_TDR, dest: MDP | LMDP | LMDP_TDR, exclude_attributes: list[str]) -> bool:
    """
    Compare two models (MDP, LMDP, or LMDP_TDR) to check if they are equal.
    The equality is defined as having the same attributes except for the excluded ones.
    
    Args:
        src (MDP | LMDP | LMDP_TDR): The source model to compare.
        dest (MDP | LMDP | LMDP_TDR): The destination model to compare.
        exclude_attributes (list[str]): List of attribute names to exclude from the comparison.
    
    Returns:
        bool: True if the models are equal, False otherwise.
    """
    type_src = type(src)
    type_dest = type(dest)
    
    if type_src != type_dest:
        return False
    
    for attr, src_val in src.__dict__.items():
        if attr in exclude_attributes: continue
        
        if not hasattr(dest, attr):
            return False
        
        dest_val = getattr(dest, attr)
        
        if isinstance(src_val, csr_matrix): src_val = src_val.toarray()
        if isinstance(dest_val, csr_matrix): dest_val = dest_val.toarray()
        
        if isinstance(src_val, np.ndarray) and isinstance(dest_val, np.ndarray):
            if not np.all(np.isclose(src_val, dest_val)):
                return False
        elif isinstance(src_val, (float, int)) and isinstance(dest_val, (float, int)):
            if not np.isclose(src_val, dest_val):
                return False
        else:
            if src_val != dest_val:
                return False
    
    return True