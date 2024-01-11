from logging import getLogger as get_logger
from typing import Callable, Iterable

import torch
import torch.distributed
from torch import Tensor

from .mezo import (
    InputType,
    ModuleType,
    average_of_updates,
    get_random_seeds,
    perturb_parameters,
    learnable_parameters,
)

logger = get_logger(__name__)


def distributed_mezo_update(
    model: ModuleType,
    inputs: InputType,
    loss_function: Callable[[ModuleType, InputType], Tensor],
    epsilon: float,
    learning_rate: float,
    base_random_seed: int,
    learnable_parameters: Callable[[ModuleType], Iterable[Tensor]] = learnable_parameters,
) -> None:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    random_seeds = get_random_seeds(base_random_seed, num_seeds=world_size)
    learning_rates = [learning_rate for _ in range(world_size)]

    random_seed = random_seeds[rank]
    logger.debug(f"Worker {rank} will use random seed of {random_seed}")

    perturb_parameters(learnable_parameters(model), epsilon, random_seed)

    with torch.no_grad():
        loss_pos = loss_function(model, inputs)

    perturb_parameters(learnable_parameters(model), -2 * epsilon, random_seed)

    with torch.no_grad():
        loss_neg = loss_function(model, inputs)

    # Reset weights to their original value.
    perturb_parameters(learnable_parameters(model), epsilon, random_seed)

    projected_grad = (loss_pos - loss_neg) / (2 * epsilon)
    logger.debug(f"Worker {rank}: local projected gradient: {projected_grad}")

    assert isinstance(projected_grad, Tensor)
    # list of Placeholder tensors, will be filled with the projected gradient from each worker.
    projected_grads: list[Tensor] = [torch.zeros_like(projected_grad) for _ in range(world_size)]
    torch.distributed.all_gather(projected_grads, projected_grad)
    if rank == 0:
        logger.debug(f"Projected gradients: {projected_grads}")

    average_of_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=learning_rates,
    )
