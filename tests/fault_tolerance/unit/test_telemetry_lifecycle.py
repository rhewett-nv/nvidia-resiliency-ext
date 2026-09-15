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

"""Tests for the launcher's telemetry lifecycle."""

import unittest
from unittest.mock import Mock, call, patch

from nvidia_resiliency_ext.fault_tolerance import ft_rendezvous_barrier as barrier
from nvidia_resiliency_ext.fault_tolerance import launcher
from nvidia_resiliency_ext.shared_utils import telemetry


class TestCycleTransitions(unittest.TestCase):
    def test_retry_closes_cycle_before_waiting_and_uses_synchronized_round(self):
        for rank, outcome in ((4, "standby"), (-1, "standby"), (None, "peer_restart")):
            with self.subTest(rank=rank):
                state = barrier._RendezvousBarrierState(Mock(), "rdzv", False)
                events = Mock()
                state._agent = Mock()
                state._rdzv_span = Mock()
                events.attach_mock(state._agent._open_telemetry_cycle, "open")
                events.attach_mock(state._agent._close_telemetry_cycle, "close")
                events.attach_mock(state._rdzv_span.close, "end_rendezvous")
                state._cached_domain_id = "none"
                state.store.add.return_value = 1

                def wait(node):
                    events.wait()
                    state._round += 3

                state._wait_for_rendezvous_open = wait
                state._wait_for_round_done = Mock(side_effect=[(rank, 5), (0, 1)])
                state._round_fenced_compare_set = Mock()
                if rank is None:
                    state._round_fenced_compare_set.side_effect = [
                        barrier._StaleRendezvousRoundError(6, 3, "slot"),
                        None,
                        None,
                    ]
                    state._wait_for_round_done.side_effect = [(0, 1)]
                with (
                    patch.object(barrier, "get_infrastructure_rank", return_value=0),
                    patch.object(state, "_current_replacement_group_id", return_value="0"),
                    patch.object(barrier, "record_profiling_event"),
                ):
                    self.assertEqual(
                        state.perform_rendezvous(barrier._NodeDesc("node", 1, 0), 1, 5, 0),
                        (0, 1),
                    )
                attrs = {"nv.nvrx.cycle.outcome": outcome}
                if rank is not None:
                    attrs["nv.nvrx.ftl.membership"] = "late_joiner" if rank == -1 else "standby"
                self.assertEqual(
                    events.mock_calls,
                    [
                        call.end_rendezvous(),
                        call.close(),
                        call.wait(),
                        call.open(3),
                        call.end_rendezvous(),
                        call.close(attrs),
                        call.end_rendezvous(),
                        call.close(),
                        call.wait(),
                        call.open(6),
                        call.end_rendezvous(
                            {
                                "nv.nvrx.ftl.group.rank": 0,
                                "nv.nvrx.ftl.membership": "active",
                            }
                        ),
                    ],
                )

    def test_cycle_callback_uses_worker_identity(self):
        agent = Mock()
        agent._node_id = "node"
        agent._worker_group.spec.rdzv_handler.get_run_id.return_value = "rdzv"
        with patch.object(
            telemetry,
            "worker_run_attributes",
            return_value={"nv.dl.run.uuid": "attempt"},
        ) as identity:
            launcher.LocalElasticAgent._open_telemetry_cycle(agent, 3)
        identity.assert_called_once_with(3, "rdzv")
        args, kwargs = agent._cycle_phase.open.call_args
        self.assertEqual(args[2]["nv.nvrx.cycle.index"], 3)
        self.assertEqual(args[2]["nv.dl.run.uuid"], "attempt")
        self.assertEqual(kwargs, {"run_uuid": "attempt"})

    def test_worker_environment_contains_the_cycle_uuid(self):
        agent = Mock()
        agent._get_global_cycle_number.return_value = 3
        agent._infra_placement_attrs.return_value = {}
        agent._launch_budget_attrs.return_value = {}
        agent._log_line_prefix_template = None
        agent._current_cycle_info_path.return_value = None
        group = Mock(workers=[Mock(local_rank=0)])
        group.spec.args = ()
        group.spec.rdzv_handler.get_run_id.return_value = "rdzv"
        with (
            patch.object(
                telemetry,
                "worker_run_attributes",
                return_value={"nv.dl.run.uuid": "attempt"},
            ) as identity,
            patch.object(telemetry, "_INHERITED_RESOURCE_ATTRIBUTES", "nv.dl.run.uuid=stale"),
            patch.object(launcher, "record_profiling_event"),
            patch.object(launcher, "start_processes") as start,
        ):
            start.return_value.pids.return_value = {}
            launcher.LocalElasticAgent._start_workers(agent, group)
        identity.assert_called_once_with(3, "rdzv")
        env = start.call_args.kwargs["envs"][0]
        self.assertEqual(env["TORCHELASTIC_RESTART_COUNT"], "3")
        self.assertIn("nv.dl.run.uuid=attempt", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertNotIn("nv.dl.run.uuid=stale", env["OTEL_RESOURCE_ATTRIBUTES"])

    def test_shutdown_failure_still_closes_phases_and_flushes(self):
        agent = Mock()
        agent._worker_group.spec.max_restarts = 5
        agent._get_global_cycle_number.return_value = 3
        agent._shutdown.side_effect = RuntimeError("shutdown failed")
        events = Mock()
        events.attach_mock(agent._run_phase.close, "run")
        events.attach_mock(agent._cycle_phase.close, "cycle")
        with (
            patch.object(telemetry, "setup_telemetry"),
            patch.object(telemetry, "record_process_startup"),
            patch.object(telemetry, "shutdown", events.flush),
        ):
            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                launcher.LocalElasticAgent.run(agent, "trainer")
        self.assertEqual(
            events.mock_calls, [call.run(), call.cycle(), call.flush(agent._tel_handle)]
        )
