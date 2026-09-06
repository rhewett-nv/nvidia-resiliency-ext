# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock
from urllib.parse import unquote

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import core


def _parse(carrier):
    return {
        segment.split("=", 1)[0]: unquote(segment.split("=", 1)[1])
        for segment in carrier.split(",")
        if "=" in segment
    }


def _caller():
    caller = core.PersistentAsyncCaller.__new__(core.PersistentAsyncCaller)
    caller.process = None
    caller.rank = 3
    caller.queue = mock.sentinel.queue
    caller.preload_q = mock.sentinel.preload_q
    caller.comp_q = mock.sentinel.comp_q
    caller.background_worker_is_daemon = True
    caller.cpu_priority = 10
    caller.io_priority = None
    caller.cpu_shm_mode = False
    return caller


def _start_worker(carrier, start_error=None, observed=None):
    observed = {} if observed is None else observed

    class FakeProcess:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            observed["carrier"] = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
            if start_error is not None:
                raise start_error

    fake_context = mock.Mock()
    fake_context.Process = FakeProcess
    caller = _caller()
    try:
        with (
            mock.patch.dict("os.environ", {}, clear=False),
            mock.patch.object(core.mp, "get_context", return_value=fake_context),
            mock.patch.object(core.PersistentAsyncCaller, "_worker_restart_callbacks", []),
        ):
            if carrier is None:
                os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
            else:
                os.environ["OTEL_RESOURCE_ATTRIBUTES"] = carrier
            try:
                caller._start_worker(3)
            finally:
                observed["restored"] = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
                observed["restored_present"] = "OTEL_RESOURCE_ATTRIBUTES" in os.environ
    finally:
        caller.process = None
    return observed


def test_worker_start_uses_live_trainer_resource_and_preserves_rank():
    trainer = (
        "nv.dl.run.uuid=run-1,nv.dl.job.uuid=job-1,nv.dl.rank=7,"
        "nv.dl.role=trainer,service.instance.id=trainer-7"
    )
    observed = _start_worker(trainer)
    worker = _parse(observed["carrier"])

    assert worker["nv.dl.run.uuid"] == "run-1"
    assert worker["nv.dl.job.uuid"] == "job-1"
    assert worker["nv.dl.rank"] == "7"
    assert worker["nv.dl.role"] == "ckpt_worker"
    assert worker["service.instance.id"] == "nvrx-ckpt3"
    assert observed["restored"] == trainer


def test_worker_start_fills_missing_rank_without_trainer_carrier():
    observed = _start_worker(None)
    worker = _parse(observed["carrier"])

    assert worker["nv.dl.rank"] == "3"
    assert worker["nv.dl.role"] == "ckpt_worker"
    assert observed["restored"] is None
    assert not observed["restored_present"]


def test_worker_start_restores_trainer_resource_when_process_start_fails():
    trainer = "nv.dl.run.uuid=run-1,nv.dl.rank=3,nv.dl.role=trainer"
    observed = {}
    with pytest.raises(RuntimeError, match="Process.start failed"):
        _start_worker(trainer, RuntimeError("Process.start failed"), observed)

    assert observed["restored"] == trainer
