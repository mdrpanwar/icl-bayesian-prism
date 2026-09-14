import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from munch import Munch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from checkpointing import (
    load_training_state,
    restore_rng_state,
    save_training_state,
)
from curriculum import Curriculum
from samplers import GaussianSampler


def curriculum_args():
    schedule = lambda start, end: Munch(
        start=start, end=end, inc=1, interval=2
    )
    return Munch(
        dims=schedule(1, 3),
        points=schedule(2, 4),
        max_freq=Munch(start=None, end=None, inc=None, interval=None),
        rff_dim=Munch(start=None, end=None, inc=None, interval=None),
    )


class CheckpointingTests(unittest.TestCase):
    def test_rng_tensors_are_moved_to_cpu_before_restoration(self):
        cpu_state = torch.zeros(1, dtype=torch.uint8)
        cuda_cpu_state = torch.ones(1, dtype=torch.uint8)
        sampler_cpu_state = torch.full((1,), 2, dtype=torch.uint8)
        torch_state = Mock()
        cuda_state = Mock()
        sampler_state = Mock()
        torch_state.cpu.return_value = cpu_state
        cuda_state.cpu.return_value = cuda_cpu_state
        sampler_state.cpu.return_value = sampler_cpu_state
        sampler = Munch(data_rand_gen=Mock())
        training_state = {
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch_state,
                "cuda": [cuda_state],
                "data_sampler": sampler_state,
            }
        }

        with (
            patch("checkpointing.torch.set_rng_state") as set_cpu_rng,
            patch("checkpointing.torch.cuda.is_available", return_value=True),
            patch("checkpointing.torch.cuda.set_rng_state_all") as set_cuda_rng,
        ):
            self.assertTrue(restore_rng_state(training_state, sampler))

        self.assertIs(set_cpu_rng.call_args.args[0], cpu_state)
        self.assertIs(set_cuda_rng.call_args.args[0][0], cuda_cpu_state)
        self.assertIs(
            sampler.data_rand_gen.set_state.call_args.args[0],
            sampler_cpu_state,
        )
        torch_state.cpu.assert_called_once_with()
    def test_full_state_round_trip_restores_exact_next_random_draws(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        curriculum = Curriculum(curriculum_args())
        sampler = GaussianSampler(2, data_seed=14)

        loss = model(torch.ones(1, 2)).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        curriculum.update()
        sampler.sample_xs(3, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.pt"
            save_training_state(
                path,
                model,
                optimizer,
                3,
                lr_scheduler=scheduler,
                curriculum=curriculum,
                data_sampler=sampler,
            )
            self.assertTrue(path.is_file())
            self.assertFalse(any(path.parent.glob(".state.pt.tmp-*")))

            expected_python = random.random()
            expected_numpy = np.random.rand()
            expected_torch = torch.rand(4)
            expected_data = sampler.sample_xs(3, 2)

            random.random()
            np.random.rand()
            torch.rand(7)

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.9)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, step_size=1, gamma=0.1
            )
            restored_curriculum = Curriculum(curriculum_args())
            restored_sampler = GaussianSampler(2, data_seed=999)

            starting_step, state = load_training_state(
                path,
                restored_model,
                restored_optimizer,
                lr_scheduler=restored_scheduler,
                curriculum=restored_curriculum,
            )
            self.assertEqual(starting_step, 4)
            self.assertTrue(restore_rng_state(state, restored_sampler))
            self.assertEqual(restored_curriculum.step_count, curriculum.step_count)
            self.assertEqual(restored_curriculum.n_points, curriculum.n_points)
            self.assertEqual(
                restored_optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["lr"],
            )
            for left, right in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.equal(left, right))

            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(torch.equal(restored_sampler.sample_xs(3, 2), expected_data))

    def test_legacy_state_remains_loadable(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        curriculum = Curriculum(curriculum_args())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step": 6,
                },
                path,
            )
            starting_step, state = load_training_state(
                path,
                model,
                optimizer,
                curriculum=curriculum,
            )

        self.assertEqual(starting_step, 7)
        self.assertFalse(restore_rng_state(state))
        self.assertEqual(curriculum.step_count, 7)


if __name__ == "__main__":
    unittest.main()
