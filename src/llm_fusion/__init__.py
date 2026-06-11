__version__ = "0.1.0"
__all__ = [
    "Match",
    "TokenMatcher",
    "Fuser",
    "patch_ouro_model",
    "generate",
    "format_hrm_prompt",
    "strip_hrm_output",
    "OURO_EOS_ID",
    "HRM_EOS_ID",
]

from llm_fusion.token_matcher import Match, TokenMatcher
from llm_fusion.fusion import Fuser
from llm_fusion.generate import (
    patch_ouro_model,
    generate,
    format_hrm_prompt,
    strip_hrm_output,
    OURO_EOS_ID,
    HRM_EOS_ID,
)
