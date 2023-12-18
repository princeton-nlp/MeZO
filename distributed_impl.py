from .simple_impl import (
    mezo_update_step,
    reconstruct_mezo_updates,
    get_device,
    average_of_mezo_updates,
    perturb_parameters,
)

# To read (according to David 'Yohan(?)') Self-Align, limitations of prompt-tuning.
# Improving language plasticity with forgetting
import torch
import torch.distributed
import torch
from torch import nn, Tensor, Parameter
from typing import Callable, TypeVar, Sequence
from torch import Generator

InputType = TypeVar("InputType")
ModelType = TypeVar("ModelType", bound=nn.Module)


def distributed_mezo_update(
    model: ModelType,
    inputs: InputType,
    loss_function: Callable[[ModelType, InputType], Tensor],
    epsilon: float,
    learning_rate: float,
    random_seed: int,
):
    is_master = torch.distributed.get_rank() == 0
    world_size = torch.distributed.get_world_size()

    if is_master:
        # Create and broadcast the random seeds to be used by each worker.
        random_seeds = torch.arange(random_seed, random_seed + world_size)
        torch.distributed.broadcast(random_seeds, src=0)
    else:
        # random_seed =
        ...

    perturb_parameters(model, epsilon, random_seed)

    with torch.no_grad():
        loss_pos = loss_function(model, inputs)

    perturb_parameters(model, -2 * epsilon, random_seed)

    with torch.no_grad():
        loss_neg = loss_function(model, inputs)

    projected_grad = (loss_pos - loss_neg) / (2 * epsilon)

    projected_grads: list[Tensor] = []
    torch.distributed.all_gather(projected_grads, projected_grad)

    average_of_mezo_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=[learning_rate for _ in projected_grads],
    )
