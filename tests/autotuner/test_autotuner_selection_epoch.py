"""AutoTuner.selection_epoch: the token callers use to memoize choose_one."""

import torch

from flashinfer.autotuner import AutoTuner, TuningConfig, autotune

from .utils import DummyRunner, reset_autotuner


def _key(shape=(8, 16)):
    return AutoTuner._get_cache_key(
        "dummy", DummyRunner(), (torch.Size(shape),), TuningConfig()
    )


def test_selection_epoch_stable_without_state_changes():
    tuner = reset_autotuner()
    epoch = tuner.selection_epoch()
    assert epoch is not None
    # Lookups (hits and misses) do not change the state choose_one reads.
    tuner.choose_one("dummy", [DummyRunner()], TuningConfig(), [torch.zeros(8, 16)])
    tuner.search_cache("dummy", [DummyRunner()], (torch.Size([8, 16]),), TuningConfig())
    assert tuner.selection_epoch() == epoch


def test_selection_epoch_changes_on_winner_cache_writes():
    tuner = reset_autotuner()
    key = _key()

    epoch = tuner.selection_epoch()
    tuner.profiling_cache[key] = (1, None)
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    tuner.profiling_cache[key] = (2, None)  # replacing a value counts too
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    del tuner.profiling_cache[key]
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    tuner.profiling_cache.update({key: (3, None)})
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    tuner.profiling_cache.pop(key)
    assert tuner.selection_epoch() != epoch


def test_selection_epoch_changes_on_loaded_configs_and_clear():
    tuner = reset_autotuner()
    key = _key()

    epoch = tuner.selection_epoch()
    tuner._file_configs[key.file_key] = ("DummyRunner", 3)
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    tuner.clear_cache()
    assert tuner.selection_epoch() != epoch

    epoch = tuner.selection_epoch()
    tuner._managed_cache = None  # attaching or detaching a managed store
    assert tuner.selection_epoch() != epoch


def test_selection_epoch_changes_when_tuning_records_a_winner(monkeypatch):
    tuner = reset_autotuner()
    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, tuning_config, **kw: float(tactic + 1),
    )
    epoch = tuner.selection_epoch()
    with autotune(True):
        tuner.choose_one("dummy", [DummyRunner()], TuningConfig(), [torch.zeros(8, 16)])
    assert tuner.selection_epoch() not in (None, epoch)


def test_selection_epoch_is_none_while_selections_must_not_be_memoized(monkeypatch):
    tuner = reset_autotuner()
    assert tuner.selection_epoch() is not None

    with autotune(True):
        assert tuner.selection_epoch() is None
    with autotune(False, tuning_buckets=(1, 2, 4)):
        assert tuner.selection_epoch() is None
    with autotune(False, skip_ops={"dummy"}):
        assert tuner.selection_epoch() is None
    # A plain non-tuning context overrides nothing.
    with autotune(False):
        assert tuner.selection_epoch() is not None
    assert tuner.selection_epoch() is not None

    tuner.is_tuning_mode = True
    try:
        assert tuner.selection_epoch() is None
    finally:
        tuner.is_tuning_mode = False

    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    assert tuner.selection_epoch() is None
