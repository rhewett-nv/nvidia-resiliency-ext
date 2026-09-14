# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the optional nemo-lens telemetry shim.

These cover the contract NVRx depends on: instrumentation is inert when
nemo-lens is missing or uninitialized, and never propagates a telemetry
failure into the workload. They run with or without nemo-lens installed.
"""

import ast
import os
import pathlib
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock
from urllib.parse import unquote

from nvidia_resiliency_ext.shared_utils import telemetry


@unittest.skipUnless(telemetry._AVAILABLE, "requires nemo-lens")
class TestCycleRunIdentity(unittest.TestCase):
    def test_exported_cycles_match_worker_resources(self):
        # Real providers in a fresh process: do not leak SDK globals into the
        # existing no-op tests. Execute the launcher's actual callback methods
        # without importing its Linux/GPU runtime dependencies on a test host.
        code = r'''
import ast, asyncio, os, pathlib, subprocess, sys, json, time
from types import SimpleNamespace
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from nemo.lens import setup_telemetry as lens_setup
from nvidia_resiliency_ext.shared_utils import telemetry

os.environ.update(SLURM_JOB_ID='123', SLURM_CLUSTER_NAME='test',
                  NEMO_LENS_ENABLED='true', NEMO_LENS_METRICS_ENABLED='false',
                  NEMO_LENS_SPAN_GROUPS='all', OTEL_RESOURCE_ATTRIBUTES='nv.dl.run.uuid=stale')
exporter = InMemorySpanExporter()
telemetry._setup_telemetry = lambda config, **kw: lens_setup(config, span_exporter=exporter, **kw)
handle = telemetry.setup_telemetry('nvrx.ft_launcher', 'agent', derive_run_uuid=False)
with telemetry.span('nvrx.ft', 'init'):
    pass
path = pathlib.Path(telemetry.__file__).parents[1] / 'fault_tolerance' / 'launcher.py'
tree = ast.parse(path.read_text())
methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
           and n.name in ('_open_telemetry_cycle', '_close_telemetry_cycle')]
namespace = {'telemetry': telemetry}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
agent = SimpleNamespace(_cycle_phase=telemetry.Phase(), _node_id='node',
    _worker_group=SimpleNamespace(spec=SimpleNamespace(
        rdzv_handler=SimpleNamespace(get_run_id=lambda: 'rdzv'))))
agent._open_telemetry_cycle = lambda count: namespace['_open_telemetry_cycle'](agent, count)
agent._close_telemetry_cycle = lambda: namespace['_close_telemetry_cycle'](agent)
barrier_path = path.with_name('ft_rendezvous_barrier.py')
barrier_tree = ast.parse(barrier_path.read_text())
perform = next(n for n in ast.walk(barrier_tree)
               if isinstance(n, ast.FunctionDef) and n.name == 'perform_rendezvous')
loop = next(n for n in perform.body if isinstance(n, ast.While))
prefix = []
for statement in loop.body:
    if isinstance(statement, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == '_rendezvous_start_time'
        for t in statement.targets):
        break
    prefix.append(statement)
round_entry = compile(ast.Module(body=prefix, type_ignores=[]), str(barrier_path), 'exec')
barrier = SimpleNamespace(_agent=agent, _round=-1, _rdzv_span=telemetry.ManualSpan())
stage_ns = {'self': barrier, 'node_desc': 'node', 'span': telemetry.span,
            'record_profiling_event': lambda *a, **k: None,
            'ProfilingEvent': SimpleNamespace(AWAIT_ROUND_STARTED=0, AWAIT_ROUND_COMPLETED=1)}
worker_code = """
import json
from nemo.lens import NemoLensConfig, setup_telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
e = InMemorySpanExporter()
h = setup_telemetry(NemoLensConfig(enabled=True, metrics_enabled=False), span_exporter=e)
with h.tracer.start_as_current_span('worker'): pass
trace.get_tracer_provider().force_flush()
print(json.dumps(dict(e.get_finished_spans()[0].resource.attributes)))
h.shutdown()
"""
uuids = []
for restart_count in (0, 3):  # skipped rounds must not be replaced with a local counter
    # Execute the actual barrier round-entry statements. The count changes
    # inside wait, so opening before synchronization would fail this test.
    barrier._wait_for_rendezvous_open = lambda node: setattr(barrier, '_round', restart_count)
    exec(round_entry, stage_ns)
    run_phase = telemetry.Phase()
    run_phase.open('nvrx.ft', 'nv.nvrx.ftl.run', {'run.phase': True})
    identity = telemetry.worker_run_attributes(restart_count, 'rdzv')
    run_uuid = identity['nv.dl.run.uuid']
    uuids.append(run_uuid)
    @telemetry.trace_fn('nvrx.ft', 'decorated')
    def work():
        with telemetry.span('nvrx.ft', 'nested'): pass
    work()
    @telemetry.trace_fn('nvrx.ft', 'async')
    async def async_work():
        with telemetry.span('nvrx.ft', 'async_nested'): pass
    asyncio.run(async_work())
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copy_context().run, work).result()
    telemetry.backdated_span('nvrx.ft', 'backdated', time.time()-1, time.time())
    try:
        with telemetry.span('nvrx.ft', 'error'):
            raise ValueError('test')
    except ValueError:
        pass
    run_phase.close()
    namespace['_close_telemetry_cycle'](agent)
    trace.get_tracer_provider().force_flush()
    cycle = exporter.get_finished_spans()
    root = next(s for s in cycle if s.name == 'nv.nvrx.ftl.cycle_start'
                and s.attributes['nv.nvrx.cycle.index'] == restart_count)
    assert root.start_time == root.end_time
    closing = next(s for s in cycle if s.name == 'nv.nvrx.ftl.cycle'
                   and s.attributes['nv.nvrx.cycle.index'] == restart_count)
    assert closing.parent == root.context
    run_marker = next(s for s in cycle if s.name == 'nv.nvrx.ftl.run_start'
                      and s.attributes['nv.dl.run.uuid'] == run_uuid)
    run_closing = next(s for s in cycle if s.name == 'nv.nvrx.ftl.run'
                       and s.attributes['nv.dl.run.uuid'] == run_uuid)
    assert run_marker.start_time == run_marker.end_time
    assert run_marker.parent == root.context
    assert run_marker.attributes['run.phase'] is True
    assert run_closing.parent == run_marker.context
    for s in cycle:
        assert 'nv.dl.run.uuid' not in s.resource.attributes
        if s.context.trace_id == root.context.trace_id or s.attributes.get('nv.dl.run.uuid') == run_uuid:
            assert s.attributes['nv.dl.run.uuid'] == run_uuid
    assert telemetry._CYCLE_RUN_UUID.get() is None
    carrier = telemetry.extended_resource_attributes(identity)
    # Execute the production worker-carrier assignments, not a test-only map.
    start_workers = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef) and n.name == '_start_workers')
    assignments = [n for n in start_workers.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in
                           ('worker_resource_attrs', 'cohort_env') for t in n.targets)]
    agent._infra_placement_attrs = lambda: {}
    agent._launch_budget_attrs = lambda: {}
    worker_ns = {'self': agent, 'telemetry': telemetry, 'restart_count': restart_count,
                 'spec': agent._worker_group.spec}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), worker_ns)
    carrier = worker_ns['cohort_env']['OTEL_RESOURCE_ATTRIBUTES']
    env = dict(os.environ, OTEL_RESOURCE_ATTRIBUTES=carrier,
               TORCHELASTIC_RESTART_COUNT=str(restart_count))
    trainer = json.loads(subprocess.check_output([sys.executable, '-c', worker_code], env=env, text=True))
    assert trainer['nv.dl.run.uuid'] == run_uuid
    checkpoint_carrier = telemetry._extend_resource_attribute_value(
        carrier, {'service.name': 'nvrx.ckpt_worker'}, True)
    env['OTEL_RESOURCE_ATTRIBUTES'] = checkpoint_carrier
    ckpt = json.loads(subprocess.check_output([sys.executable, '-c', worker_code], env=env, text=True))
    assert ckpt['nv.dl.run.uuid'] == run_uuid
    with telemetry.span('nvrx.ft', 'await_round'): pass
assert uuids[0] != uuids[1]
trace.get_tracer_provider().force_flush()
for s in exporter.get_finished_spans():
    if s.name in ('init', 'await_round'):
        assert 'nv.dl.run.uuid' not in s.attributes
        assert 'nv.dl.run.uuid' not in s.resource.attributes
handle.shutdown()
'''
        for scenario in ("slurm", "array", "local"):
            with self.subTest(scenario=scenario):
                variant = code
                insertion = ""
                if scenario == "array":
                    insertion = (
                        "os.environ.update(SLURM_ARRAY_JOB_ID='100', SLURM_ARRAY_TASK_ID='2')\n"
                    )
                elif scenario == "local":
                    insertion = "os.environ.pop('SLURM_JOB_ID', None)\n"
                variant = variant.replace(
                    "exporter = InMemorySpanExporter()",
                    insertion + "exporter = InMemorySpanExporter()",
                    1,
                )
                result = subprocess.run(
                    [sys.executable, "-c", variant], capture_output=True, text=True, timeout=45
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class TestTelemetryIsInert(unittest.TestCase):
    """Instrumentation must be a no-op before/without setup_telemetry()."""

    def test_cycle_identity_without_lens(self):
        with unittest.mock.patch.object(telemetry, "_AVAILABLE", False):
            self.assertEqual(telemetry.worker_run_attributes(3, "rdzv"), {})
            phase = telemetry.Phase()
            phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", run_uuid="unused")
            self.assertIsNone(telemetry._CYCLE_RUN_UUID.get())
            phase.close()
            handle = telemetry.setup_telemetry("nvrx.ft_launcher", derive_run_uuid=False)
            handle.shutdown()

    def test_managed_span_yields_and_runs_body(self):
        ran = False
        with telemetry.span("nvrx.ft", "nv.nvrx.ftl.cycle") as active:
            ran = True
            self.assertIsNone(active)
        self.assertTrue(ran)

    def test_managed_span_propagates_body_exceptions(self):
        # Telemetry must never swallow a workload error -- notably SignalException,
        # which torch elastic raises out of the launcher's monitor loop.
        with self.assertRaises(ValueError):
            with telemetry.span("nvrx.ft", "nv.nvrx.ftl.cycle"):
                raise ValueError("from the instrumented body")

    def test_managed_span_accepts_attributes(self):
        with telemetry.span("nvrx.ckpt", "nv.nvrx.ckpt.save.request", {"nv.nvrx.ckpt.call_idx": 7}):
            pass

    def test_trace_fn_returns_a_working_decorator(self):
        @telemetry.trace_fn("nvrx.ft", "nv.nvrx.ftl.worker_launch")
        def start(a, b=2):
            return a + b

        self.assertEqual(start(1), 3)
        self.assertEqual(start(1, b=10), 11)

    def test_trace_fn_propagates_exceptions(self):
        @telemetry.trace_fn("nvrx.ft", "nv.nvrx.ftl.teardown")
        def boom():
            raise RuntimeError("worker teardown failed")

        with self.assertRaises(RuntimeError):
            boom()

    def test_set_span_attributes_without_active_span(self):
        telemetry.set_span_attributes({"nv.nvrx.cycle.index": 3, "nv.nvrx.ftl.node": "node-0"})


class TestManualSpan(unittest.TestCase):
    """ManualSpan must tolerate every order the launcher can call it in."""

    def test_all_methods_are_safe_before_open(self):
        span = telemetry.ManualSpan()
        span.set({"nv.nvrx.cycle.index": 0})
        span.close({"nv.nvrx.cycle.outcome": "terminated"})
        span.close()

    def test_close_is_idempotent(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        span.close({"nv.nvrx.cycle.outcome": "completed"})
        span.close()
        span.close({"nv.nvrx.cycle.outcome": "terminated"})

    def test_reopen_closes_the_previous_span(self):
        # The restart path relies on this: a cycle is left open so teardown lands
        # inside it, and the next rendezvous closes it by opening the next cycle.
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        first_stack = span._stack
        span.set({"nv.nvrx.cycle.outcome": "failed"})
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 1})
        self.assertIsNot(span._stack, first_stack)
        span.close()
        self.assertIsNone(span._stack)

    def test_set_tolerates_none_and_empty(self):
        span = telemetry.ManualSpan()
        span.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        span.set(None)
        span.set({})
        span.close()


class TestMarkAndFlush(unittest.TestCase):

    def test_mark_is_inert(self):
        telemetry.mark("nvrx.ft", "nv.nvrx.ftl.fault")
        telemetry.mark(
            "nvrx.ft",
            "nv.nvrx.ftl.fault",
            {"nv.nvrx.ftl.cycle.state": "FAILED", "nv.nvrx.ftl.cycle.failures": 2},
        )

    def test_flush_is_inert(self):
        # Must tolerate a provider with no force_flush (the no-op one) and a
        # provider that was never configured at all.
        telemetry.flush()
        telemetry.flush(timeout_ms=1)

    def test_shutdown_is_bounded_and_never_raises(self):
        class SlowHandle:
            def __init__(self):
                self.entered = threading.Event()

            def shutdown(self, timeout_ms: int = 5000):
                self.entered.set()
                time.sleep(30)  # a collector that is gone

        handle = SlowHandle()
        started = time.monotonic()
        telemetry.shutdown(handle, timeout_s=0.2)
        elapsed = time.monotonic() - started
        self.assertTrue(handle.entered.wait(1), "shutdown() was never called")
        self.assertLess(elapsed, 5, "shutdown was not bounded")


class TestBackdatedSpan(unittest.TestCase):
    """Startup windows are reconstructed from timestamps, so guard the inputs."""

    def test_inert_without_telemetry(self):
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.startup", 1000.0, 1016.7)
        telemetry.backdated_span(
            "job", "nv.nvrx.ftl.python.imports", 1016.7, 1020.9, {"nv.nvrx.ftl.node": "n0"}
        )

    def test_absent_timestamps_are_dropped_not_raised(self):
        # SLURM_JOB_START_TIME is absent off Slurm, so the caller passes None
        # rather than pre-checking; `end > None` would be a TypeError.
        telemetry.backdated_span("job", "pre_startup", None, 1016.7)
        telemetry.backdated_span("job", "pre_startup", 1000.0, None)
        telemetry.backdated_span("job", "pre_startup", None, None)

    def test_non_positive_window_is_dropped(self):
        # A coarse clock can make a fast window measure as zero-length or inverted.
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.imports", 1016.7, 1000.0)
        telemetry.backdated_span("job", "nv.nvrx.ftl.python.imports", 1000.0, 1000.0)


class TestExtendedResourceAttributes(unittest.TestCase):
    """The agent extends a variable it must never parse, once per cohort."""

    def extend(self, inherited, attributes):
        with unittest.mock.patch.object(telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", inherited):
            return telemetry.extended_resource_attributes(attributes)

    def test_carries_the_inherited_value_through_untouched(self):
        # Whatever the launching environment set is opaque here, including keys
        # NVRx has no notion of.
        inherited = "slurm.job_id=370487,cluster=oci-aga,job.uid=b3f1"
        result = self.extend(inherited, {"nv.nvrx.cycle.index": 2})
        self.assertTrue(result.startswith(inherited + ","))
        self.assertTrue(result.endswith("nv.nvrx.cycle.index=2"))

    def test_works_with_nothing_inherited(self):
        self.assertEqual(self.extend("", {"nv.nvrx.cycle.index": 0}), "nv.nvrx.cycle.index=0")

    def test_no_attributes_leaves_the_value_alone(self):
        self.assertEqual(self.extend("cluster=oci-aga", {}), "cluster=oci-aga")
        self.assertEqual(self.extend("", {}), "")

    def test_values_are_percent_encoded(self):
        # A value containing a comma or an equals would otherwise be read back as
        # extra pairs, silently rewriting the resource.
        result = self.extend("", {"nv.nvrx.ftl.membership": "active,standby=maybe"})
        self.assertEqual(result, "nv.nvrx.ftl.membership=active%2Cstandby%3Dmaybe")

    def test_extends_the_inherited_value_not_the_last_one(self):
        # The agent relaunches a cohort every cycle. Extending its own previous
        # output would append another nv.nvrx.cycle.index each time, without bound.
        inherited = "cluster=oci-aga"
        first = self.extend(inherited, {"nv.nvrx.cycle.index": 0})
        second = self.extend(inherited, {"nv.nvrx.cycle.index": 1})
        self.assertEqual(first.count("nv.nvrx.cycle.index"), 1)
        self.assertEqual(second.count("nv.nvrx.cycle.index"), 1)
        self.assertEqual(second, "cluster=oci-aga,nv.nvrx.cycle.index=1")

    def test_default_ignores_a_later_environment_value(self):
        with (
            unittest.mock.patch.object(
                telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", "job.uid=imported"
            ),
            unittest.mock.patch.dict(
                "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": "job.uid=live"}, clear=False
            ),
        ):
            result = telemetry.extended_resource_attributes({"nv.nvrx.cycle.index": 2})
        self.assertIn("job.uid=imported", result)
        self.assertNotIn("job.uid=live", result)


class TestPublishResourceAttributes(unittest.TestCase):
    """The only channel that reaches a spawned child, so it has to be exact."""

    def setUp(self):
        self.env = unittest.mock.patch.dict(
            "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": "job.uid=abc"}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.inherited = unittest.mock.patch.object(
            telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", "job.uid=abc"
        )
        self.inherited.start()
        self.addCleanup(self.inherited.stop)

    def test_the_child_sees_both_inherited_and_published(self):
        with telemetry.publish_resource_attributes({"nv.dl.rank": 3}):
            published = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        self.assertIn("job.uid=abc", published)
        self.assertIn("nv.dl.rank=3", published)

    def test_the_parent_is_restored(self):
        # Left set, it would describe this process and every later child of it.
        with telemetry.publish_resource_attributes({"nv.dl.rank": 3}):
            pass
        self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "job.uid=abc")

    def test_restored_even_when_the_spawn_raises(self):
        with self.assertRaises(RuntimeError):
            with telemetry.publish_resource_attributes({"nv.dl.rank": 3}):
                raise RuntimeError("Process.start() failed")
        self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "job.uid=abc")

    def test_an_unset_variable_is_removed_again_not_left_empty(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
            with unittest.mock.patch.object(telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", ""):
                with telemetry.publish_resource_attributes({"nv.dl.rank": 3}):
                    self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "nv.dl.rank=3")
                self.assertNotIn("OTEL_RESOURCE_ATTRIBUTES", os.environ)

    def test_successive_publishes_do_not_accumulate(self):
        # A worker restarts per cycle; building from the last value grows unbounded.
        for _ in range(3):
            with telemetry.publish_resource_attributes({"nv.dl.rank": 3}):
                published = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        self.assertEqual(published.count("nv.dl.rank"), 1)

    @staticmethod
    def _parse(carrier):
        return {
            segment.split("=", 1)[0]: unquote(segment.split("=", 1)[1])
            for segment in carrier.split(",")
            if "=" in segment
        }

    def test_fallback_live_base_and_mixed_precedence_preserve_valid_rank(self):
        trainer = (
            "nv.dl.run.uuid=run-1,nv.dl.job.uuid=job-1,nv.dl.rank=7,"
            "nv.dl.role=trainer,service.instance.id=trainer-7"
        )
        with (
            unittest.mock.patch.object(telemetry, "_AVAILABLE", False),
            unittest.mock.patch.dict(
                "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": trainer}, clear=False
            ),
        ):
            with telemetry.publish_resource_attributes(
                {
                    "nv.dl.role": "ckpt_worker",
                    "service.instance.id": "nvrx-ckpt3",
                },
                use_current=True,
                fill_missing={"nv.dl.rank": 3},
            ):
                published = self._parse(os.environ["OTEL_RESOURCE_ATTRIBUTES"])

            self.assertEqual(published["nv.dl.run.uuid"], "run-1")
            self.assertEqual(published["nv.dl.job.uuid"], "job-1")
            self.assertEqual(published["nv.dl.rank"], "7")
            self.assertEqual(published["nv.dl.role"], "ckpt_worker")
            self.assertEqual(published["service.instance.id"], "nvrx-ckpt3")
            self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], trainer)

    def test_fallback_discards_malformed_rank_before_filling_it(self):
        trainer = "nv.dl.rank,nv.dl.role=trainer"
        with (
            unittest.mock.patch.object(telemetry, "_AVAILABLE", False),
            unittest.mock.patch.dict(
                "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": trainer}, clear=False
            ),
        ):
            with telemetry.publish_resource_attributes(
                {"nv.dl.role": "ckpt_worker"},
                use_current=True,
                fill_missing={"nv.dl.rank": 3},
            ):
                carrier = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
                published = self._parse(carrier)

            self.assertNotIn("nv.dl.rank", carrier.split(","))
            self.assertEqual(published["nv.dl.rank"], "3")
            self.assertEqual(published["nv.dl.role"], "ckpt_worker")
            self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], trainer)

    def test_lens_encoder_receives_each_precedence_layer_in_order(self):
        trainer = "nv.dl.run.uuid=run-1,nv.dl.role=trainer," "service.instance.id=trainer-3"
        intermediate = (
            "nv.dl.run.uuid=run-1,nv.dl.role=trainer," "service.instance.id=trainer-3,nv.dl.rank=3"
        )
        worker = (
            "nv.dl.run.uuid=run-1,nv.dl.rank=3,nv.dl.role=ckpt_worker,"
            "service.instance.id=nvrx-ckpt3"
        )
        worker_attributes = {
            "nv.dl.role": "ckpt_worker",
            "service.instance.id": "nvrx-ckpt3",
        }
        lens_encoder = unittest.mock.Mock(side_effect=[intermediate, worker])

        with (
            unittest.mock.patch.object(telemetry, "_AVAILABLE", True),
            unittest.mock.patch.object(
                telemetry, "_extend_resource_attributes", lens_encoder, create=True
            ),
            unittest.mock.patch.dict(
                "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": trainer}, clear=False
            ),
        ):
            with telemetry.publish_resource_attributes(
                worker_attributes,
                use_current=True,
                fill_missing={"nv.dl.rank": 3},
            ):
                self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], worker)

            self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], trainer)

        self.assertEqual(
            lens_encoder.call_args_list,
            [
                unittest.mock.call(trainer, {"nv.dl.rank": 3}, overwrite=False),
                unittest.mock.call(intermediate, worker_attributes, overwrite=True),
            ],
        )

    def test_live_base_fills_rank_when_the_trainer_carrier_has_none(self):
        trainer = "nv.dl.run.uuid=run-1,nv.dl.role=trainer"
        with unittest.mock.patch.dict(
            "os.environ", {"OTEL_RESOURCE_ATTRIBUTES": trainer}, clear=False
        ):
            with telemetry.publish_resource_attributes(
                {"nv.dl.role": "ckpt_worker"},
                use_current=True,
                fill_missing={"nv.dl.rank": 3},
            ):
                published = self._parse(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertEqual(published["nv.dl.rank"], "3")
        self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "job.uid=abc")


class TestPhase(unittest.TestCase):
    """A phase is a mark now and a backdated span later; check the two line up.

    nemo-lens and the OTel SDK are optional and usually absent here, so the two
    primitives a phase is built from are replaced and the phase's own logic --
    span naming, the backdated window, where attributes land -- is what is under
    test.
    """

    def setUp(self):
        self.marks = []
        self.spans = []

        def fake_mark(group, name, attributes=None):
            self.marks.append((group, name, attributes))
            return f"ctx-of-{name}"

        def fake_backdated(group, name, start, end, attributes=None, parent=None):
            self.spans.append((group, name, start, end, attributes, parent))

        for target, replacement in (("mark", fake_mark), ("backdated_span", fake_backdated)):
            patcher = unittest.mock.patch.object(telemetry, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_marks_the_start_and_backdates_the_span_to_it(self):
        phase = telemetry.Phase()
        before = time.time()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 2})
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        after = time.time()

        self.assertEqual(
            self.marks, [("nvrx.ft", "nv.nvrx.ftl.cycle_start", {"nv.nvrx.cycle.index": 2})]
        )
        group, name, start, end, attributes, parent = self.spans[0]
        self.assertEqual((group, name), ("nvrx.ft", "nv.nvrx.ftl.cycle"))
        # The span covers the window, rather than being an instant at close.
        self.assertLessEqual(before, start)
        self.assertLessEqual(start, end)
        self.assertLessEqual(end, after)
        # Same trace as the mark, so the spans that ran inside the phase join it.
        self.assertEqual(parent, "ctx-of-nv.nvrx.ftl.cycle_start")
        # Opening attributes carry through to the span; close adds to them.
        self.assertEqual(
            attributes, {"nv.nvrx.cycle.index": 2, "nv.nvrx.cycle.outcome": "completed"}
        )

    def test_open_attributes_go_on_the_mark_and_the_span(self):
        # The mark is the only record while the phase runs, and the span is the only
        # one a consumer filters. Both need the attributes.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 3})
        phase.close()
        self.assertEqual(self.marks[0][2], {"nv.nvrx.cycle.index": 3})
        self.assertEqual(self.spans[0][4], {"nv.nvrx.cycle.index": 3})

    def test_close_attributes_override_opening_ones(self):
        # Lets a required attribute be seeded at open rather than on each close path.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.ftl.membership": "unjoined"})
        phase.close({"nv.nvrx.ftl.membership": "standby"})
        self.assertEqual(self.marks[0][2], {"nv.nvrx.ftl.membership": "unjoined"})
        self.assertEqual(self.spans[0][4], {"nv.nvrx.ftl.membership": "standby"})

    def test_set_accumulates_until_close(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.set({"nv.nvrx.ftl.group.rank": 3})
        phase.set({"nv.nvrx.ftl.membership": "active"})
        phase.close({"nv.nvrx.cycle.outcome": "failed"})
        self.assertEqual(
            self.spans[0][4],
            {
                "nv.nvrx.ftl.group.rank": 3,
                "nv.nvrx.ftl.membership": "active",
                "nv.nvrx.cycle.outcome": "failed",
            },
        )

    def test_close_is_idempotent(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close()
        phase.close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(len(self.spans), 1)

    def test_close_without_open_is_a_no_op(self):
        telemetry.Phase().close({"nv.nvrx.cycle.outcome": "completed"})
        self.assertEqual(self.spans, [])

    def test_open_closes_the_previous_phase(self):
        # The launcher reuses one handle across cycles and relies on this.
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 0})
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle", {"nv.nvrx.cycle.index": 1})
        self.assertEqual(len(self.spans), 1, "the first cycle was never emitted")
        self.assertEqual(len(self.marks), 2)

    def test_attributes_do_not_leak_between_phases(self):
        phase = telemetry.Phase()
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close({"nv.nvrx.cycle.outcome": "failed"})
        phase.open("nvrx.ft", "nv.nvrx.ftl.cycle")
        phase.close()
        self.assertEqual(self.spans[1][4], {})


class TestSetupTelemetry(unittest.TestCase):

    def test_returns_handle_with_idempotent_shutdown(self):
        # Disabled is the default (NEMO_LENS_ENABLED is unset), so this exercises
        # the no-op path whether or not nemo-lens is installed.
        handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
        self.assertTrue(hasattr(handle, "shutdown"))
        handle.shutdown()
        handle.shutdown()

    def test_init_failure_does_not_propagate(self):
        original = telemetry._AVAILABLE
        telemetry._AVAILABLE = True
        try:
            # _NemoLensConfig is undefined when nemo-lens is absent; either way the
            # shim must degrade to a no-op handle rather than raise into the caller.
            with unittest.mock.patch.object(
                telemetry, "_setup_telemetry", side_effect=RuntimeError("boom"), create=True
            ):
                with unittest.mock.patch.object(
                    telemetry, "_NemoLensConfig", create=True
                ) as config_cls:
                    config_cls.from_env.return_value = unittest.mock.MagicMock()
                    handle = telemetry.setup_telemetry("nvrx.test", "nvrx-test0")
            self.assertIsInstance(handle, telemetry._NoOpHandle)
        finally:
            telemetry._AVAILABLE = original


@unittest.skipUnless(telemetry._AVAILABLE, "nemo-lens is not installed")
class TestSpanGroupRegistration(unittest.TestCase):
    """The NVRx groups must be selectable, or every NVRx span is dark.

    nemo-lens ships no group names, so importing the shim is what makes these
    resolvable -- at import, not in ``setup_telemetry``, which the trainer never calls.
    """

    def test_registered_under_the_nvrx_namespace(self):
        from nemo.lens import SpanRegistry

        self.assertIn(telemetry._NAMESPACE, SpanRegistry.namespaces())

    def test_every_group_resolves_by_name(self):
        from nemo.lens import SpanRegistry

        for group in telemetry._GROUPS:
            enabled, pending = SpanRegistry.resolve(group)
            self.assertEqual(enabled, frozenset([group]))
            self.assertEqual(pending, frozenset(), f"{group!r} resolved to nothing")

    def test_presets_resolve_to_their_members(self):
        from nemo.lens import SpanRegistry

        for preset, members in telemetry._PRESETS.items():
            enabled, _ = SpanRegistry.resolve(preset)
            self.assertTrue(
                members <= enabled, f"preset {preset!r} is missing {sorted(members - enabled)}"
            )

    def test_phases_are_a_drill_down_not_a_default(self):
        # Per-request spans are always on; per-stage ones are opted into.
        self.assertIn(telemetry._CKPT, telemetry._PRESETS["default"])
        self.assertNotIn(telemetry._CKPT_PHASES, telemetry._PRESETS["default"])
        self.assertIn(telemetry._CKPT_PHASES, telemetry._PRESETS["per_step"])


class TestEverySpanGroupIsRegistered(unittest.TestCase):
    """No call site may name a group NVRx does not register.

    nemo-lens reports an unregistered group and carries on, which is right for a
    job-wide spec and wrong for a call site, where it is a typo that costs those
    spans silently. Reads the source rather than importing it, so it runs lens-free.
    """

    #: Every call that takes a span group as its first positional argument.
    _CALLS = frozenset(
        ["span", "linked_span", "mark", "trace_fn", "backdated_span", "record_process_startup"]
    )

    def test_no_call_site_names_an_unregistered_group(self):
        root = pathlib.Path(telemetry.__file__).parent.parent
        offenders = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name not in self._CALLS:
                    continue
                group = node.args[0]
                if not isinstance(group, ast.Constant) or not isinstance(group.value, str):
                    continue
                if group.value not in telemetry._GROUPS:
                    offenders.append(
                        f"{path.relative_to(root)}:{group.lineno} {name}({group.value!r})"
                    )
        self.assertEqual(offenders, [], "call sites naming an unregistered span group")

    def test_the_scan_actually_finds_call_sites(self):
        # Guards the test above against passing because it matched nothing.
        root = pathlib.Path(telemetry.__file__).parent.parent
        found = sum(
            1
            for path in root.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.Call)
            and node.args
            and (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None)
            )
            in self._CALLS
            and isinstance(node.args[0], ast.Constant)
        )
        self.assertGreater(found, 10, "the span-group scan matched almost nothing")


if __name__ == "__main__":
    unittest.main()
