import copy
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import torch.distributed
from torch import Tensor, nn

from mezo import mezo_update_step
from mezo.distributed_mezo import distributed_mezo_update
from mezo.mezo import ModelType, get_random_seeds


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


def run_single_distributed_update_and_save_resulting_weights(
    model: ModelType,
    inputs: Tensor,
    loss_function: Callable[[ModelType, Tensor], Tensor],
    epsilon: float,
    learning_rate: float,
    base_random_seed: int,
    rank: int,
    world_size: int,
    save_final_weights_path: Path,
):
    """Used as the target of a multiprocessing.Process for test purposes.

    Performs a single update step, then writes the resulting weights to the given path.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[{rank+1}/{world_size}] " + "%(asctime)s %(levelname)s %(message)s",
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12323"
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing the process group in worker {rank+1}/{world_size}")

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch.distributed.init_process_group(
        backend="gloo" if world_size > device_count else "nccl",
        init_method="env://",
        timeout=timedelta(seconds=30),
        rank=rank,
        world_size=world_size,
    )
    logger.info(f"Done initializing the process group in worker {rank+1}/{world_size}")

    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Done initializing the process group in worker {rank+1}/{world_size}")

    # Maybe overly cautious, but I don't know if the model's weights are shared or not across
    # processes.
    # model.load_state_dict(copy.deepcopy(model.state_dict()))
    model = model.to(device)
    distributed_mezo_update(
        model,
        inputs=inputs.to(device),
        loss_function=loss_function,
        epsilon=epsilon,
        learning_rate=learning_rate,
        base_random_seed=base_random_seed,
    )
    logger.info(f"Saving weights to {save_final_weights_path}")
    torch.save(model.state_dict(), save_final_weights_path)


# device_count = torch.cuda.device_count()
@pytest.mark.parametrize(
    "n_workers",
    [2],
)
def test_distributed_mezo(
    model: nn.Module,
    inputs: Tensor,
    seed: int,
    tmp_path: Path,
    device: torch.device,
    n_workers: int,
):
    """IDEA: Create two processes, make each of them run a single step of distributed_mezo_update.

    - check that have the same weights at the end.
    """
    initial_weights = copy.deepcopy(model.state_dict())  # save for later.
    # model = model.to("cpu")
    # inputs = inputs.to("cpu")

    epsilon: float = 0.1
    learning_rate: float = 0.1

    processes: list[torch.multiprocessing.Process] = []
    # Slightly change the inputs between workers to simulate different batches.
    worker_inputs = [inputs.clone() + rank for rank in range(n_workers)]
    # worker_seeds = [seed + rank for rank in range(n_workers)]
    worker_final_weights_paths = [
        tmp_path / f"worker_{rank}_final_weights.pth" for rank in range(n_workers)
    ]

    torch.multiprocessing.set_start_method("forkserver")

    base_seed = seed

    for rank, (worker_input, worker_final_weights_path) in enumerate(
        zip(worker_inputs, worker_final_weights_paths)
    ):
        process = torch.multiprocessing.Process(
            target=run_single_distributed_update_and_save_resulting_weights,
            kwargs=dict(
                model=model,
                inputs=worker_input,
                loss_function=loss_function,
                epsilon=epsilon,
                learning_rate=learning_rate,
                base_random_seed=base_seed,
                rank=rank,
                world_size=n_workers,
                save_final_weights_path=worker_final_weights_path,
            ),
            daemon=True,
        )
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)

    for process in processes:
        if process.is_alive():
            process.kill()

    # Check that all workers ended up with the same weights at the end:
    assert worker_final_weights_paths[0].exists()
    first_worker_final_weights = torch.load(worker_final_weights_paths[0])
    for final_weights_path in worker_final_weights_paths[1:]:
        assert final_weights_path.exists()
        worker_final_weights = torch.load(final_weights_path)
        torch.testing.assert_close(worker_final_weights, first_worker_final_weights)

    # Check that this final weight is indeed the average of the weights of all workers.
    new_weights_list: list[dict[str, Tensor]] = []

    for rank, (worker_input, worker_seed) in enumerate(
        zip(worker_inputs, get_random_seeds(base_seed, num_seeds=n_workers))
    ):
        model.load_state_dict(initial_weights)
        mezo_update_step(
            model,
            worker_input,
            loss_function,
            epsilon=epsilon,
            learning_rate=learning_rate,
            random_seed=worker_seed,
        )
        new_weights_list.append(copy.deepcopy(model.state_dict()))

    average_of_weights = {
        key: torch.mean(torch.stack([w[key] for w in new_weights_list]), dim=0)
        for key in new_weights_list[0].keys()
    }

    torch.testing.assert_close(first_worker_final_weights, average_of_weights)
