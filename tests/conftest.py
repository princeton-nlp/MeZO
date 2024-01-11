import numpy as np
import pytest
import torch
from torch import Tensor, nn


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def inputs(device: torch.device, seed: int) -> Tensor:
    batch_size = 32
    return torch.randn(
        batch_size,
        10,
        generator=torch.Generator(device).manual_seed(seed),
        device=device,
    )


@pytest.fixture
def model(device: torch.device, inputs: Tensor, seed: int):
    dims = int(np.prod(inputs.shape[1:]))
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims, 16, device=device),
            nn.Tanh(),
            nn.Linear(16, dims, device=device),
        )
        model(inputs)  # instantiate weights.
    return model


def loss_function(model: nn.Module, inputs: torch.Tensor) -> Tensor:
    # Simple dumb reconstruction loss.
    return torch.nn.functional.mse_loss(model(inputs), inputs)
