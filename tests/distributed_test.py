import copy
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.distributed
from torch import Tensor, nn

from mezo import (
    mezo_update_step,
    average_of_mezo_updates,
    get_random_seeds,
    distributed_mezo_update,
)
from mezo.mezo import ModuleType
from .conftest import loss_function

logger = logging.getLogger(__name__)


def run_single_distributed_update_and_save_resulting_weights(
    model: ModuleType,
    inputs: Tensor,
    loss_function: Callable[[ModuleType, Tensor], Tensor],
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

    # just to make sure that we're not modifying any shared weight in any way.
    model.load_state_dict(copy.deepcopy(model.state_dict()))

    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[{rank+1}/{world_size}] " + "%(asctime)s %(levelname)s %(message)s",
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12323"
    logger = logging.getLogger(__name__)

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Initializing the process group in worker {rank+1}/{world_size}")
    torch.distributed.init_process_group(
        # NOTE: nccl doesn't allow multiple workers on the same GPU.
        backend="gloo" if world_size > device_count else "nccl",
        init_method="env://",
        timeout=timedelta(seconds=30),
        rank=rank,
        world_size=world_size,
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
    n_workers: int,
):
    """IDEA: Create two processes, make each of them run a single step of distributed_mezo_update.

    - check that have the same weights at the end.
    - check that this final weight is the average of the weights if each update had been done
      separately.
    """
    initial_weights = copy.deepcopy(model.state_dict())  # save for later.

    epsilon: float = 0.1
    learning_rate: float = 0.1

    processes: list[torch.multiprocessing.Process] = []

    # Slightly change the inputs between workers to simulate different batches.
    worker_inputs = [inputs.clone() + rank for rank in range(n_workers)]
    worker_final_weights_paths = [
        tmp_path / f"worker_{rank}_final_weights.pth" for rank in range(n_workers)
    ]

    torch.multiprocessing.set_start_method("spawn")

    base_seed = seed

    for rank, (worker_input, worker_final_weights_path) in enumerate(
        zip(worker_inputs, worker_final_weights_paths)
    ):
        process = torch.multiprocessing.Process(
            target=run_single_distributed_update_and_save_resulting_weights,
            kwargs=dict(
                model=copy.deepcopy(model),
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
    # Each worker has the same final weight.
    for final_weights_path in worker_final_weights_paths[1:]:
        assert final_weights_path.exists()
        worker_final_weights = torch.load(final_weights_path)
        torch.testing.assert_close(worker_final_weights, first_worker_final_weights)

    # Check that the final weights are indeed the average of the updates from each worker.
    projected_grads: list[Tensor] = []

    random_seeds = get_random_seeds(base_seed, num_seeds=n_workers)
    logger.info(f"Random seeds: {random_seeds}")
    weights_after_each_worker_update: list[dict[str, Tensor]] = []
    for rank, (worker_input, worker_seed) in enumerate(zip(worker_inputs, random_seeds)):
        model.load_state_dict(initial_weights)
        worker_projected_grad = mezo_update_step(
            model,
            worker_input,
            loss_function,
            epsilon=epsilon,
            learning_rate=learning_rate,
            random_seed=worker_seed,
        )
        projected_grads.append(worker_projected_grad)
        weights_after_each_worker_update.append(copy.deepcopy(model.state_dict()))

    # Manually check that it matches the average of the weights if each update had been done
    # separately.
    average_of_weights = {
        key: torch.mean(torch.stack([w[key] for w in weights_after_each_worker_update]), dim=0)
        for key in weights_after_each_worker_update[0].keys()
    }
    torch.testing.assert_close(first_worker_final_weights, average_of_weights)

    # Same check, but using the `average_of_mezo_updates` function.
    # NOTE: There's also a separate test for that function.
    model.load_state_dict(initial_weights)
    average_of_mezo_updates(
        model,
        random_seeds=random_seeds,
        projected_grads=projected_grads,
        learning_rates=[learning_rate for _ in range(n_workers)],
    )
    torch.testing.assert_close(first_worker_final_weights, model.state_dict())
