import copy

import torch
import torch.distributed
from torch import Tensor, nn

from mezo import mezo_update, reconstruct_mezo_updates, average_of_mezo_updates
from .conftest import loss_function


def test_mezo_update_step_reproducible(model: nn.Module, inputs: torch.Tensor, seed: int):
    epsilon: float = 0.1
    learning_rate: float = 0.1

    initial_model_weights = copy.deepcopy(model.state_dict())

    projected_grad = mezo_update(
        model,
        inputs,
        loss_function,
        epsilon=epsilon,
        learning_rate=learning_rate,
        random_seed=seed,
    )
    new_weights = copy.deepcopy(model.state_dict())

    # Assert that the weights are modified:
    for name, weight in new_weights.items():
        assert not torch.allclose(weight, initial_model_weights[name])

    # Reset the model weights.
    model.load_state_dict(initial_model_weights)

    same_projected_grad = mezo_update(
        model,
        inputs,
        loss_function,
        epsilon=epsilon,
        learning_rate=learning_rate,
        random_seed=seed,
    )
    same_new_weights = copy.deepcopy(model.state_dict())
    torch.testing.assert_close(projected_grad, same_projected_grad)
    torch.testing.assert_close(new_weights, same_new_weights)

    # Reset the model weights.
    model.load_state_dict(initial_model_weights)

    # Different random seed should lead to different weights.
    other_projected_grad = mezo_update(
        model,
        inputs,
        loss_function,
        epsilon=epsilon,
        learning_rate=learning_rate,
        random_seed=seed + 1,
    )
    different_weights = copy.deepcopy(model.state_dict())
    for name, weight in new_weights.items():
        assert not torch.allclose(weight, different_weights[name])
    assert not torch.allclose(projected_grad, other_projected_grad)


def test_reconstruct_mezo_updates(model: nn.Module, inputs: Tensor, seed: int):
    """Test that `reconstruct_mezo_updates` is the same as using `mezo_update_step` multiple
    times."""
    projected_grads = []
    learning_rate = 0.1
    n_steps = 3

    random_seeds = [seed + i for i in range(n_steps)]
    learning_rates = [learning_rate for _ in range(n_steps)]
    epsilon = 0.1

    initial_weights = copy.deepcopy(model.state_dict())
    for step, seed in enumerate(random_seeds):
        projected_grad = mezo_update(
            model,
            inputs,  # use same inputs for each step, but it could also change, doesn't matter.
            loss_function,
            epsilon=epsilon,
            learning_rate=learning_rate,
            random_seed=seed,
        )
        projected_grads.append(projected_grad)

    new_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(initial_weights)

    reconstruct_mezo_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=learning_rates,
    )

    same_new_weights = copy.deepcopy(model.state_dict())
    torch.testing.assert_close(new_weights, same_new_weights)


def test_average_of_mezo_updates(model: nn.Module, inputs: Tensor, seed: int):
    projected_grads = []
    learning_rate = 0.1
    n_steps = 3

    random_seeds = [seed + i for i in range(n_steps)]
    learning_rates = [learning_rate for _ in range(n_steps)]
    epsilon = 0.1

    initial_weights = copy.deepcopy(model.state_dict())

    new_weights_list: list[dict[str, Tensor]] = []

    for seed in random_seeds:
        # NOTE: Use `assign=False` explicitly just to make sure we don't change the initial weights
        model.load_state_dict(initial_weights, assign=False)
        projected_grad = mezo_update(
            model,
            inputs,  # using the same input for each step but it could also change, doesn't matter.
            loss_function,
            epsilon=epsilon,
            learning_rate=learning_rate,
            random_seed=seed,
        )
        projected_grads.append(projected_grad)

        new_weights = copy.deepcopy(model.state_dict())
        new_weights_list.append(new_weights)

    average_of_new_weights = {
        key: torch.mean(torch.stack([w[key] for w in new_weights_list]), dim=0)
        for key in new_weights_list[0].keys()
    }

    model.load_state_dict(initial_weights)
    average_of_mezo_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=learning_rates,
    )

    weights_after_average_update = copy.deepcopy(model.state_dict())

    torch.testing.assert_close(weights_after_average_update, average_of_new_weights)
