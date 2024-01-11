from typing import Callable, Iterable, TypeVar

import torch
from torch import Generator, Tensor, nn

InputType = TypeVar("InputType")
ModuleType = TypeVar("ModuleType", bound=nn.Module)


def learnable_parameters(model: nn.Module) -> Iterable[Tensor]:
    return (p for p in model.parameters() if p.requires_grad)


def update(
    model: ModuleType,
    inputs: InputType,
    loss_function: Callable[[ModuleType, InputType], Tensor],
    epsilon: float,
    learning_rate: float,
    random_seed: int,
    learnable_parameters: Callable[[ModuleType], Iterable[Tensor]] = learnable_parameters,
) -> Tensor:
    perturb_parameters(learnable_parameters(model), epsilon, random_seed)

    with torch.no_grad():
        loss_pos = loss_function(model, inputs)

    perturb_parameters(learnable_parameters(model), -2 * epsilon, random_seed)

    with torch.no_grad():
        loss_neg = loss_function(model, inputs)

    projected_grad = (loss_pos - loss_neg) / (2 * epsilon)

    # reset the model parameters
    perturb_parameters(learnable_parameters(model), epsilon, random_seed)

    update_params(
        learnable_parameters(model),
        random_seed=random_seed,
        projected_grad=projected_grad,
        learning_rate=learning_rate,
    )
    return projected_grad


def perturb_parameters(parameters: Iterable[Tensor], epsilon: float, rng_seed: int) -> None:
    device: torch.device | None = None
    rng_gen: torch.Generator | None = None
    for param in parameters:
        if device is None:
            device = param.device
        if rng_gen is None:
            rng_gen = torch.Generator(device=device).manual_seed(rng_seed)
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
    parameters: Iterable[Tensor],
    random_seed: int,
    learning_rate: float,
    projected_grad: Tensor,
):
    """Update the parameters using the projected gradient and the random seed."""
    device: torch.device | None = None
    rng_gen: torch.Generator | None = None
    for param in parameters:
        if device is None:
            device = param.device
        if rng_gen is None:
            rng_gen = torch.Generator(device=device).manual_seed(random_seed)
        update_param(
            param,
            rng_gen=rng_gen,
            learning_rate=learning_rate,
            projected_grad=projected_grad,
        )


def update_param(
    param: Tensor,
    *,
    rng_gen: Generator,
    learning_rate: float,
    projected_grad: Tensor | float,
    extra_coefficient: float = 1.0,
    max_params_at_a_time: int | None = 100_000,
) -> None:
    """Updates a parameter in-place using the coefficients and the projected grad value.

    Parameters
    ----------
    param: the parameter tensor to update in-place.
    rng_gen: Torch random number generator
    learning_rate: learning rate for the update
    projected_grad: The projected gradient computed as the (loss_pos - loss_neg) / (2 * epsilon)
    extra_coefficient: An extra coefficient applied to the update. Useful when averaging updates.
    max_params_at_a_time: Maximum number of parameters to update at a time, to limit the GPU \
        memory usage.
    """
    if max_params_at_a_time is None or param.numel() <= max_params_at_a_time:
        z = torch.randn(
            param.shape,
            dtype=param.dtype,
            device=param.device,
            generator=rng_gen,
        )
        param.data.subtract_(extra_coefficient * learning_rate * projected_grad * z)
    else:
        # IDEA: Split up the update, doing it row by row instead, whenever the weight is too large.
        # This is to save some GPU memory.

        # NOTE: The RNG should still be fine, because the same number of samples are drawn every
        # time for that particular weight.
        for param_row in param:
            update_param(
                param_row,
                rng_gen=rng_gen,
                learning_rate=learning_rate,
                projected_grad=projected_grad,
                extra_coefficient=extra_coefficient,
                max_params_at_a_time=max_params_at_a_time,
            )


def reconstruct_updates(
    model: ModuleType,
    random_seeds: list[int],
    projected_grads: list[Tensor],
    learning_rates: list[float],
    learnable_parameters: Callable[[ModuleType], Iterable[Tensor]] = learnable_parameters,
) -> None:
    """Recover the final weights given the projected grads and the random seeds at each step."""
    device: torch.device | None = None
    rng_generators: list[torch.Generator] | None = None
    for param in learnable_parameters(model):
        if device is None or rng_generators is None:
            device = param.device
            rng_generators = [
                torch.Generator(device=device).manual_seed(random_seed)
                for random_seed in random_seeds
            ]

        for learning_rate, projected_grad, rng_gen in zip(
            learning_rates, projected_grads, rng_generators
        ):
            update_param(
                param,
                rng_gen=rng_gen,
                learning_rate=learning_rate,
                projected_grad=projected_grad,
            )


def average_of_updates(
    model: ModuleType,
    random_seeds: list[int],
    projected_grads: list[Tensor],
    learning_rates: list[float],
    learnable_parameters: Callable[[ModuleType], Iterable[Tensor]] = learnable_parameters,
) -> None:
    """Perform the average of multiple MeZO updates given the projected grads and the random seeds.

    This is useful for the distributed implementation of MeZO.
    """
    assert len(random_seeds) == len(projected_grads) == len(learning_rates)
    N = len(random_seeds)

    device: torch.device | None = None
    rng_generators: list[torch.Generator] | None = None
    for param in learnable_parameters(model):
        if device is None or rng_generators is None:
            device = param.device
            rng_generators = [
                torch.Generator(device=device).manual_seed(random_seed)
                for random_seed in random_seeds
            ]

        for learning_rate, projected_grad, rng_gen in zip(
            learning_rates, projected_grads, rng_generators
        ):
            update_param(
                param,
                rng_gen=rng_gen,
                learning_rate=learning_rate,
                projected_grad=projected_grad,
                extra_coefficient=1 / N,
            )


def get_random_seeds(base_random_seed: int, num_seeds: int) -> list[int]:
    """Returns a sequence of integers to be used as random seeds for each worker or step."""
    return torch.randint(
        0,
        int(1e8),
        (num_seeds,),
        generator=torch.Generator().manual_seed(base_random_seed),
    ).tolist()
