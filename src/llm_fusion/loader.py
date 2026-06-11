"""Model loading and initialization for Ouro-1.4B and HRM-Text-1B."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class CausalLM(Protocol):
    def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object: ...


def patch_ouro_model(config: object) -> None:
    config._attn_implementation = "eager"  # type: ignore[attr-defined]


def load_models(
    base_dir: str | Path = "",
    ouro_path: str = "ByteDance/Ouro-1.4B",
    hrm_path: str = "sapientinc/HRM-Text-1B",
    model: str = "fused",
    local: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[CausalLM | None, CausalLM | None, str]:
    """Load one or both models. Returns (ouro_model, hrm_model, device)."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        print(f"Error: requires torch and transformers ({e})", file=sys.stderr)
        sys.exit(1)

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    log = logging.getLogger(__name__)
    if debug:
        log.setLevel(logging.DEBUG)
    elif verbose:
        log.setLevel(logging.INFO)

    log.info("Device: %s, dtype: %s", device, dtype)

    load_ouro = model in ("fused", "ouro")
    load_hrm = model in ("fused", "hrm")

    ouro_model: CausalLM | None = None
    hrm_model: CausalLM | None = None

    print(f"Loading models on {device}...", file=sys.stderr)

    if load_ouro:
        path = str(bd / "Ouro-1.4B") if local else ouro_path
        log.info("Loading Ouro model from %s", path)
        ouro_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        log.debug("Ouro config._attn_implementation set to 'eager'")
        ouro_model = AutoModelForCausalLM.from_pretrained(
            path,
            config=ouro_config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        log.info("Ouro model loaded")

    if load_hrm:
        path = str(bd / "HRM-Text-1B") if local else hrm_path
        log.info("Loading HRM model from %s", path)
        hrm_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map=device,
        )
        log.info("HRM model loaded")

    return ouro_model, hrm_model, device
