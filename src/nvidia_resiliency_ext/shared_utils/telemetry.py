# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Optional nemo-lens OTel instrumentation. The only file in NVRx that imports it.

Every export is a no-op when nemo-lens is absent or its span group is off, so
callers need no guards. Attributes are dicts everywhere because NVRx attribute
names are dotted and cannot be Python keywords.

Design and rationale: docs/design/telemetry/NEMO_LENS.md.
"""

import inspect
import logging
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)
_CYCLE_RUN_UUID = ContextVar("nvrx_cycle_run_uuid", default=None)
_RUN_UUID = "nv.dl.run.uuid"

#: Span groups NVRx emits, and the presets selecting them. nvrx.ckpt is one span
#: per checkpoint request per side; nvrx.ckpt.phases breaks each into its stages
#: and is opt-in, being per-stage cardinality.
_NAMESPACE = "nvrx"
_JOB = "nvrx.job"
_FT = "nvrx.ft"
_CKPT = "nvrx.ckpt"
_CKPT_PHASES = "nvrx.ckpt.phases"
_GROUPS = frozenset([_JOB, _FT, _CKPT, _CKPT_PHASES])
_PRESETS = {
    "default": frozenset([_JOB, _FT, _CKPT]),
    "per_step": frozenset([_JOB, _FT, _CKPT, _CKPT_PHASES]),
    "profiling": _GROUPS,
}

# Captured before anything can extend it. Extensions build from this, never from
# the last extension, or a relaunched cohort accumulates a key per restart.
_INHERITED_RESOURCE_ATTRIBUTES = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")

try:
    # Underscored names exist only when nemo-lens is installed. trace_fn's alias
    # is the PEP 484 re-export form, marking a name this module never calls.
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanRegistry as _SpanRegistry
    from nemo.lens import get_tracer as _get_tracer
    from nemo.lens import is_span_group_enabled as _is_span_group_enabled
    from nemo.lens import managed_span as _managed_span
    from nemo.lens import safe_set_span_attributes as _safe_set_span_attributes
    from nemo.lens import setup_telemetry as _setup_telemetry
    from nemo.lens.resources import extend_otel_resource_attributes as _extend_resource_attributes
    from nemo.lens.resources.slurm import derive_nv_dl_run_uuid as _derive_run_uuid
    from nemo.lens.span_utilities import emit_span as _emit_span

    _AVAILABLE = True

except ImportError:
    # nemo-lens absent, or a version whose surface moved: either way, nothing to call.
    _AVAILABLE = False

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _AVAILABLE = False


if _AVAILABLE:
    try:
        # At import, not in setup_telemetry: the trainer emits NVRx checkpoint
        # spans and never calls setup_telemetry, so its groups would be dark.
        _SpanRegistry.register(_NAMESPACE, _GROUPS, _PRESETS)
    except Exception:
        # A name collision costs these groups, not all telemetry.
        logger.warning("Could not register the NVRx span groups", exc_info=True)


if not _AVAILABLE:

    @contextmanager
    def _managed_span(group, name, tracer=None, **attributes):
        """No-op stand-in for ``nemo.lens.managed_span``."""
        yield None


class _NoOpHandle:
    """Stand-in for ``nemo.lens.TelemetryHandle`` when telemetry is unavailable."""

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass


def setup_telemetry(
    service_name: str,
    instance_id: Optional[str] = None,
    resource_attributes: Optional[dict] = None,
    *,
    derive_run_uuid: bool = True,
):
    """Initialize nemo-lens. Call once, at process start, only in a process NVRx owns.

    ``service_name`` becomes ``service.name``, overriding ``OTEL_SERVICE_NAME``,
    which names the workload rather than these processes. ``instance_id`` becomes
    ``service.instance.id``; omit it when a parent published one through
    :func:`publish_resource_attributes`. One of the two must supply it -- nemo-lens
    derives its own from ``nv.dl.rank``, which no NVRx process has a usable value for.
    """
    if not _AVAILABLE:
        return _NoOpHandle()
    try:
        config = _NemoLensConfig.from_env()
        config.service_name = service_name
        attributes = {"service.instance.id": instance_id} if instance_id else {}
        attributes.update(resource_attributes or {})
        options = {} if derive_run_uuid else {"derive_run_uuid": False}
        return _setup_telemetry(config, resource_attributes=attributes, **options)
    except Exception:
        logger.warning("nemo-lens init failed, continuing without telemetry", exc_info=True)
        return _NoOpHandle()


def shutdown(handle, timeout_s: float = 2.0) -> None:
    """Flush and shut down, bounded.

    ``TelemetryHandle.shutdown()`` can block for the exporter's whole retry budget
    against a collector that is gone, which outlasts a SIGTERM grace period.
    """
    worker = threading.Thread(target=handle.shutdown, daemon=True)
    worker.start()
    worker.join(timeout_s)


def flush(timeout_ms: int = 1500) -> None:
    """Export what is buffered, for a point where this process may be killed next."""
    if not _AVAILABLE:
        return
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=timeout_ms)


class ManualSpan:
    """A span opened in one call and closed in another. No-op while nothing is open.

    ORDERING CONTRACT, from ``contextvars``: open() and close() must run on the same
    thread, and anything opened after this one must close before it does. Violating
    either does not raise -- it silently restores a stale context, parenting later
    spans to a span that has ended.
    """

    def __init__(self) -> None:
        self._stack: Optional[ExitStack] = None
        self._span = None

    def open(self, group: str, name: str, attributes: Optional[dict] = None) -> None:
        """Start a span, closing any span this handle already had open."""
        self.close()
        self._stack = ExitStack()
        self._span = self._stack.enter_context(span(group, name, attributes))

    def set(self, attributes: Optional[dict] = None) -> None:
        """Set attributes on the open span."""
        if self._span is None or not attributes:
            return
        for key, value in attributes.items():
            self._span.set_attribute(key, value)

    def close(self, attributes: Optional[dict] = None) -> None:
        """Set any final attributes and end the span. Idempotent."""
        self.set(attributes)
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._span = None


def _cycle_attributes(attributes=None):
    attrs = dict(attributes or {})
    run_uuid = _CYCLE_RUN_UUID.get()
    if run_uuid is not None:
        attrs[_RUN_UUID] = run_uuid
    return attrs


def worker_run_attributes(restart_count: int, rendezvous_run_id: str) -> dict:
    """Resolve identity with the exact restart count sent to trainer workers."""
    if not _AVAILABLE:
        return {}
    env = dict(os.environ)
    env["TORCHELASTIC_RESTART_COUNT"] = str(restart_count)
    env["TORCHELASTIC_RUN_ID"] = rendezvous_run_id
    run_uuid = _derive_run_uuid(env, run_id=rendezvous_run_id)
    return {_RUN_UUID: run_uuid} if run_uuid is not None else {}


def trace_fn(group, name, tracer=None):
    """Decorate a span with cycle identity, without work before the group gate."""

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped(*args, **kwargs):
                if not _AVAILABLE or not _is_span_group_enabled(group):
                    return await func(*args, **kwargs)
                active_tracer = tracer if tracer is not None else _get_tracer("nemo.lens")
                with _managed_span(group, name, tracer=active_tracer, **_cycle_attributes()):
                    return await func(*args, **kwargs)

            return async_wrapped

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not _AVAILABLE or not _is_span_group_enabled(group):
                return func(*args, **kwargs)
            active_tracer = tracer if tracer is not None else _get_tracer("nemo.lens")
            with _managed_span(group, name, tracer=active_tracer, **_cycle_attributes()):
                return func(*args, **kwargs)

        return wrapped

    return decorator


def span(group: str, name: str, attributes: Optional[dict] = None):
    """A span around a block, yielding it (or None when the group is off).

    Dict adapter over ``managed_span``, which takes keywords: a dotted attribute
    name can never be one. Returns the upstream context manager unwrapped.
    """
    if _AVAILABLE and _is_span_group_enabled(group):
        attributes = _cycle_attributes(attributes)
    return _managed_span(group, name, **(attributes or {}))


def backdated_span(
    group: str,
    name: str,
    start: Optional[float],
    end: Optional[float],
    attributes: Optional[dict] = None,
    parent=None,
):
    """Record a span for a window that elapsed before there was a tracer.

    ``start`` and ``end`` are wall-clock seconds; ``parent`` is usually the
    ``SpanContext`` of the ``mark`` that opened the window, and without one the span
    roots its own trace. Lens owns timing, gating and completion; zero duration
    is valid. Return the completed span context when available.
    """
    if not _AVAILABLE or start is None or end is None:
        return None
    from opentelemetry import trace
    from opentelemetry.context import Context

    context = Context()
    if parent is not None:
        context = trace.set_span_in_context(trace.NonRecordingSpan(parent), context)
    completed = _emit_span(
        None,
        name,
        start,
        end,
        group=group,
        attributes=_cycle_attributes(attributes),
        context=context,
    )
    return completed.get_span_context() if completed is not None else None


def mark(group: str, name: str, attributes: Optional[dict] = None):
    """Record an instant: a zero-duration span pinning a moment in time.

    Returns its ``SpanContext``, or None when the group is off. A mark exports
    immediately, so the context outlives it -- ids, not a handle to anything live.
    """
    if not _AVAILABLE or not _is_span_group_enabled(group):
        return None
    timestamp = time.time()
    completed = _emit_span(
        None, name, timestamp, timestamp, group=group, attributes=_cycle_attributes(attributes)
    )
    return completed.get_span_context() if completed is not None else None


def set_span_attributes(attributes: dict) -> None:
    """Set attributes on the active span, for inside a ``@trace_fn``. No-op if none."""
    if not _AVAILABLE:
        return
    from opentelemetry import trace

    _safe_set_span_attributes(trace.get_current_span(), attributes)


def _extend_resource_attribute_value(base: str, attributes: dict, overwrite: bool) -> str:
    """Extend one encoded Resource carrier using Lens or the local fallback."""
    if _AVAILABLE:
        return _extend_resource_attributes(base, attributes, overwrite=overwrite)

    from urllib.parse import quote

    added = {str(key): quote(str(value), safe="") for key, value in attributes.items()}
    valid_segments = [
        (segment, segment.split("=", 1)[0].strip())
        for segment in base.split(",")
        if "=" in segment and segment.split("=", 1)[0].strip()
    ]
    existing_keys = {key for _, key in valid_segments}
    if not overwrite:
        added = {key: value for key, value in added.items() if key not in existing_keys}
    kept = [segment for segment, key in valid_segments if not overwrite or key not in added]
    return ",".join(kept + [f"{key}={value}" for key, value in added.items()])


def extended_resource_attributes(
    attributes: dict, *, use_current: bool = False, fill_missing: Optional[dict] = None
) -> str:
    """Extend an ``OTEL_RESOURCE_ATTRIBUTES`` carrier with more pairs.

    NVRx never parses the variable -- it is an opaque string to extend. Extending is
    from the value inherited at import by default, never from the last extension, or
    a relaunched cohort accumulates a key per cycle. ``use_current`` is reserved for
    a child-process boundary that must inherit Resource identity published after this
    module was imported. ``fill_missing`` supplies compatibility defaults without
    replacing identity already present in the selected base. ``attributes`` always
    overwrite because the publishing process is authoritative for them.
    """
    key = "OTEL_RESOURCE_ATTRIBUTES"
    base = os.environ.get(key, "") if use_current else _INHERITED_RESOURCE_ATTRIBUTES
    if fill_missing:
        base = _extend_resource_attribute_value(base, fill_missing, overwrite=False)
    return _extend_resource_attribute_value(base, attributes, overwrite=True)


@contextmanager
def publish_resource_attributes(
    attributes: dict, *, use_current: bool = False, fill_missing: Optional[dict] = None
):
    """Publish attributes into the environment, for a child spawned inside.

    ``multiprocessing.Process`` has no ``env``, so the environment at ``start()``
    is the only channel to a spawned child. Wrap that call. Restored on exit, or it
    would describe this process and every later child of it too. Values arrive in
    the child as strings.
    """
    key = "OTEL_RESOURCE_ATTRIBUTES"
    previous = os.environ.get(key)
    os.environ[key] = extended_resource_attributes(
        attributes, use_current=use_current, fill_missing=fill_missing
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def record_process_startup(
    group: str,
    imports_started: float,
    imports_finished: float,
    attributes: Optional[dict] = None,
) -> None:
    """Record how long this process took to become able to run.

    Two backdated windows: process creation to the entry module's first statement,
    and that module's top-level imports. Both root their own trace.
    """
    try:
        import psutil

        created = psutil.Process().create_time()
    except Exception:
        logger.debug("Process create time unavailable", exc_info=True)
        created = None
    backdated_span(group, "nv.nvrx.ftl.python.startup", created, imports_started, attributes)
    backdated_span(
        group, "nv.nvrx.ftl.python.imports", imports_started, imports_finished, attributes
    )


class Phase:
    """A long window, recorded as a start mark now and a backdated span later.

    For a window too long to hold a span open across, since a span exports only
    when it ends. ``open()`` marks ``<name>_start`` and makes it the active context,
    so spans on this thread nest under the phase; ``close()`` emits ``<name>``
    backdated to that mark. A phase that never closes still leaves the mark and
    everything that ran inside it.

    ``open`` attributes go on both records; ``set`` and ``close`` reach only the
    span and override by name.

    ORDERING CONTRACT, from ``contextvars``: ``open()`` and ``close()`` must run on
    the same thread, and anything opened after this one must close before it does.
    """

    def __init__(self) -> None:
        self._group: Optional[str] = None
        self._name: Optional[str] = None
        self._start: Optional[float] = None
        self._parent = None
        self._token = None
        self._attributes: dict = {}
        self._run_uuid_token = None

    def open(
        self, group: str, name: str, attributes: Optional[dict] = None, *, run_uuid=None
    ) -> None:
        """Mark the start of the phase, closing any phase this handle had open."""
        self.close()
        if run_uuid is not None and _AVAILABLE and _is_span_group_enabled(group):
            self._run_uuid_token = _CYCLE_RUN_UUID.set(run_uuid)
        self._group, self._name = group, name
        self._start = time.time()
        # Seeded, not emptied: a consumer filtering spans never sees the mark's
        # attributes, so a key left only there cannot be grouped on.
        self._attributes = dict(attributes or {})
        self._parent = mark(group, f"{name}_start", attributes)
        if not _AVAILABLE or self._parent is None:
            return
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace

            self._token = otel_context.attach(
                trace.set_span_in_context(trace.NonRecordingSpan(self._parent))
            )
        except Exception:
            # Losing the ambient context costs nesting, not spans.
            logger.debug("Could not make %s the active context", self._name, exc_info=True)

    def set(self, attributes: Optional[dict] = None) -> None:
        """Record attributes to be emitted on the backdated span at close."""
        if self._start is None or not attributes:
            return
        self._attributes.update(attributes)

    def close(self, attributes: Optional[dict] = None) -> None:
        """Emit the backdated span covering the phase. Idempotent."""
        self.set(attributes)
        if self._start is None:
            return
        if self._token is not None:
            try:
                from opentelemetry import context as otel_context

                otel_context.detach(self._token)
            except Exception:
                # A phase opened after this one outlived it, so the token is not the
                # top of the stack. The span is still correct; the next open() fixes
                # the stale ambient context.
                logger.debug("Out-of-order close for phase %s", self._name, exc_info=True)
            self._token = None
        backdated_span(
            self._group,
            self._name,
            self._start,
            time.time(),
            self._attributes,
            parent=self._parent,
        )
        self._group = self._name = self._start = self._parent = None
        self._attributes = {}
        if self._run_uuid_token is not None:
            _CYCLE_RUN_UUID.reset(self._run_uuid_token)
            self._run_uuid_token = None
