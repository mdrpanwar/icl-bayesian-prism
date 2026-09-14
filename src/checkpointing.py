"""Signal-safe, resumable training checkpoints shared by ICL and MAML."""

from __future__ import annotations

import os
from pathlib import Path
import random
import signal

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def _rng_state(data_sampler=None):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if data_sampler is not None and hasattr(data_sampler, "data_rand_gen"):
        state["data_sampler"] = data_sampler.data_rand_gen.get_state()
    return state


def restore_rng_state(training_state, data_sampler=None):
    """Restore stochastic state when present; tolerate legacy checkpoints."""
    state = training_state.get("rng_state")
    if not state:
        return False

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])
    if (
        data_sampler is not None
        and hasattr(data_sampler, "data_rand_gen")
        and "data_sampler" in state
    ):
        data_sampler.data_rand_gen.set_state(state["data_sampler"].cpu())
    return True


def _curriculum_state(curriculum):
    return {
        "step_count": curriculum.step_count,
        "n_dims_truncated": curriculum.n_dims_truncated,
        "n_points": curriculum.n_points,
        "max_freq": curriculum.max_freq,
        "rff_dim": curriculum.rff_dim,
    }


def restore_curriculum_state(curriculum, training_state, completed_steps):
    state = training_state.get("curriculum_state")
    if state:
        for name, value in state.items():
            setattr(curriculum, name, value)
        return

    # Backward compatibility for checkpoints written before curriculum state
    # was stored explicitly.
    for _ in range(completed_steps):
        curriculum.update()


def atomic_torch_save(payload, path):
    """Write on the destination filesystem, then atomically replace `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_state(
    path,
    model,
    optimizer,
    train_step,
    *,
    lr_scheduler=None,
    curriculum=None,
    data_sampler=None,
):
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": int(train_step),
        "rng_state": _rng_state(data_sampler),
    }
    if lr_scheduler is not None:
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if curriculum is not None:
        state["curriculum_state"] = _curriculum_state(curriculum)
    atomic_torch_save(state, path)


def load_training_state(
    path,
    model,
    optimizer,
    *,
    lr_scheduler=None,
    curriculum=None,
):
    """Load a state checkpoint and return `(starting_step, state)`.

    Legacy state files containing only model/optimizer/train_step remain valid.
    RNG restoration is deliberately deferred until samplers have been built.
    """
    path = Path(path)
    if not path.is_file():
        return 0, None

    device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if lr_scheduler is not None and "lr_scheduler_state_dict" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])

    starting_step = int(state["train_step"]) + 1
    if curriculum is not None:
        restore_curriculum_state(curriculum, state, starting_step)
    return starting_step, state


class PreemptionHandler:
    """Turn SIGTERM/SIGINT into a checkpoint request at the next safe step."""

    def __init__(self):
        self.signum = None
        self._previous = {}

    @property
    def requested(self):
        return self.signum is not None

    def _request(self, signum, _frame):
        if self.signum is None:
            self.signum = signum
            print(
                f"[checkpoint] received signal {signum}; "
                "saving after the current optimizer step",
                flush=True,
            )

    def install(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request)
        return self

    def restore(self):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def exit_after_checkpoint(self):
        """Exit nonzero so RunAI retries the same persistent workload."""
        if self.signum is None:
            return
        signum = self.signum
        self.restore()
        raise SystemExit(128 + int(signum))
