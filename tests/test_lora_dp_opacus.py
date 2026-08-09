"""LoRA Opacus üretim sarmalayıcı testleri."""

import torch
import torch.nn as nn

from rag.lora_dp_opacus import (
    make_private_lora,
    opacus_module_validator_available,
    prepare_model_for_opacus,
)


def test_prepare_model_for_opacus_tiny():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    fixed, errors = prepare_model_for_opacus(model, strict=False)
    assert fixed is not None
    assert isinstance(errors, list)


def test_make_private_lora_smoke():
    if not opacus_module_validator_available():
        return
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(3, 4)
    ds = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model, opt, loader, engine, ok = make_private_lora(
        model,
        opt,
        loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        secure_mode=False,
    )
    assert ok
    for batch in loader:
        y = model(batch[0])
        loss = y.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        break
    assert engine is not None
