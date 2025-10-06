"""PLL minimal artifact package.

Provides:
- runtime: get_model/get_tokenizer/_resolve_device
- geometry_helpers: attention submodule access and head metadata
- phase_group / phase_sum: minimal algebra for PLL terminology
- PAC/PLA/piecewise CLI modules
"""

__all__ = [
    "runtime",
    "geometry_helpers",
    "phase_group",
    "phase_sum",
    "piecewise_sps",
]
