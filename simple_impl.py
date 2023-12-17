import torch
from torch import nn, Tensor, Parameter
from typing import Callable, TypeVar, Sequence
from torch import Generator

InputType = TypeVar("InputType")
ModelType = TypeVar("ModelType", bound=nn.Module)


def mezo_update_step(
    model: ModelType,
    inputs: InputType,
    loss_function: Callable[[ModelType, InputType], Tensor],
    epsilon: float,
    learning_rate: float,
    random_seed: int,
) -> Tensor:
    device = get_device(model)

    perturb_parameters(model, epsilon, random_seed)

    with torch.random.fork_rng(devices=[device]), torch.no_grad():
        loss_pos = loss_function(model, inputs)

    perturb_parameters(model, -2 * epsilon, random_seed)

    with torch.random.fork_rng(devices=[device]), torch.no_grad():
        loss_neg = loss_function(model, inputs)

    projected_grad = (loss_pos - loss_neg) / (2 * epsilon)

    # reset the model parameters
    perturb_parameters(model, epsilon, random_seed)

    update_params(
        model,
        random_seed=random_seed,
        projected_grad=projected_grad,
        learning_rate=learning_rate,
    )
    return projected_grad


def perturb_parameters(model: nn.Module, epsilon: float, rng_seed: int) -> None:
    device = get_device(model)
    rng_gen = torch.Generator(device=device).manual_seed(rng_seed)
    for param in model.parameters():
        param.data.add_(
            epsilon
            * torch.randn(
                param.shape,
                dtype=param.dtype,
                device=param.device,
                generator=rng_gen,
            )
        )


def update_params(
    model: nn.Module,
    random_seed: int,
    projected_grad: Tensor,
    learning_rate: float,
):
    """Update the parameters using the projected gradient."""
    device = get_device(model)
    rng_gen = torch.Generator(device=device).manual_seed(random_seed)
    for param in model.parameters():
        z = torch.randn(
            param.shape, dtype=param.dtype, device=param.device, generator=rng_gen
        )
        param.data.subtract_(learning_rate * projected_grad * z)


def reconstruct_mezo_updates(
    model: nn.Module,
    random_seeds: list[int],
    projected_grads: list[Tensor],
    learning_rates: list[float],
):
    """Given the projected grads and the random seeds, reconstruct multiple updates."""
    device = get_device(model)
    rng_generators = [
        torch.Generator(device=device).manual_seed(random_seed)
        for random_seed in random_seeds
    ]
    for param in model.parameters():
        for learning_rate, projected_grad, rng_gen in zip(
            learning_rates, projected_grads, rng_generators
        ):
            z = torch.randn(
                param.shape, dtype=param.dtype, device=param.device, generator=rng_gen
            )
            param.data.subtract_(learning_rate * projected_grad * z)


def average_of_mezo_updates(
    model: nn.Module,
    random_seeds: list[int],
    projected_grads: list[Tensor],
    learning_rates: list[float],
):
    """Given the projected grads and the random seeds, perform an update with the average of the
    updates for each worker.
    """
    device = get_device(model)
    rng_generators = [
        torch.Generator(device=device).manual_seed(random_seed)
        for random_seed in random_seeds
    ]
    N = len(random_seeds)
    assert len(random_seeds) == len(projected_grads)
    for param in model.parameters():
        average_update = torch.zeros_like(param)

        for learning_rate, projected_grad, rng_gen in zip(
            learning_rates, projected_grads, rng_generators
        ):
            update = (
                (1 / N)
                * learning_rate
                * projected_grad
                * torch.randn(  # z
                    param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    generator=rng_gen,
                )
            )
        param.data.subtract_(average_update)


def update_param(
    param: Tensor,
    rng_gen: Generator,
    learning_rate: float,
    projected_grad: Tensor | float,
    max_params_at_a_time: int = 100_000,
):
    """IDEA: Split up the update, doing it row by row instead, whenever the weight is too large.
    This is to save some GPU memory.

    NOTE: The RNG should still be fine, because the same number of samples are drawn every time for
    that particular weight.
    """
    if max_params_at_a_time is None or param.numel() <= max_params_at_a_time:
        param.subtract_(
            learning_rate
            * projected_grad
            * torch.randn(  # z
                param.shape,
                dtype=param.dtype,
                device=param.device,
                generator=rng_gen,
            )
        )
        return
    for param_row in param:
        update_param(
            param_row,
            rng_gen=rng_gen,
            learning_rate=learning_rate,
            projected_grad=projected_grad,
            max_params_at_a_time=max_params_at_a_time,
        )


def get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
