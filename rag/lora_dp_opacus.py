"""Opacus + PEFT LoRA üretim sarmalayıcıları (gradient sample uyumu)."""

from __future__ import annotations

from typing import Any, Tuple


def opacus_module_validator_available() -> bool:
    try:
        from opacus.validators import ModuleValidator  # noqa: F401

        return True
    except Exception:
        return False


def prepare_model_for_opacus(model: Any, *, strict: bool = False) -> Tuple[Any, list]:
    """ModuleValidator ile PEFT/LoRA modülünü Opacus için düzeltir."""
    if not opacus_module_validator_available():
        return model, []
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=strict)
    if errors:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=strict)
    return model, list(errors or [])


def make_private_lora(
    module: Any,
    optimizer: Any,
    data_loader: Any,
    *,
    noise_multiplier: float,
    max_grad_norm: float,
    secure_mode: bool = False,
    grad_sample_mode: str = "hooks",
) -> Tuple[Any, Any, Any, Any, bool]:
    """PrivacyEngine.make_private — üretim ayarlarıyla."""
    from opacus import PrivacyEngine

    engine = PrivacyEngine(secure_mode=secure_mode)
    module, optimizer, loader = engine.make_private(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        grad_sample_mode=grad_sample_mode,
    )
    return module, optimizer, loader, engine, True
