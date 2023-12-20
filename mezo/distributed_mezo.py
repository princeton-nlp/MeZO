from logging import getLogger as get_logger
from typing import Callable

import torch
import torch.distributed
from torch import Tensor

from .mezo import (
    InputType,
    ModelType,
    average_of_mezo_updates,
    get_random_seeds,
    perturb_parameters,
)

# To read (according to David 'Yohan(?)') Self-Align, limitations of prompt-tuning.
# Improving language plasticity with selective forgetting

logger = get_logger(__name__)


def distributed_mezo_update(
    model: ModelType,
    inputs: InputType,
    loss_function: Callable[[ModelType, InputType], Tensor],
    epsilon: float,
    learning_rate: float,
    base_random_seed: int,
):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    random_seeds = get_random_seeds(base_random_seed, num_seeds=world_size)
    learning_rates = [learning_rate for _ in range(world_size)]
    # torch.distributed.scatter(random_seed, random_seeds, src=0)
    random_seed = random_seeds[rank]

    assert len(random_seeds) == world_size

    perturb_parameters(model, epsilon, random_seed)

    with torch.no_grad():
        loss_pos = loss_function(model, inputs)

    perturb_parameters(model, -2 * epsilon, random_seed)

    with torch.no_grad():
        loss_neg = loss_function(model, inputs)

    projected_grad = (loss_pos - loss_neg) / (2 * epsilon)
    assert isinstance(projected_grad, Tensor)
    projected_grads: list[Tensor] = [projected_grad for _ in range(world_size)]
    torch.distributed.all_gather(projected_grads, projected_grad)
    logger.debug(f"Projected gradient: {projected_grad}")
    logger.debug(f"Projected gradients: {projected_grads}")
    average_of_mezo_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=learning_rates,
    )
