"""Repo-wide pytest configuration.

Exists for one reason: **ONNX Runtime telemetry is disabled for every test process.**

`ORT_DISABLE_TELEMETRY` is read by ONNX Runtime when its native library is loaded, so it has to be
in the environment before anything imports `onnxruntime`. A fixture — even a session-scoped autouse
one — runs too late, because the suite conftests import `onnxruntime` at module scope. The rootdir
conftest is imported before those, which makes this the only place the assignment is reliably early
enough.

Beyond not phoning home from a test run, this also avoids a real failure mode. ORT's 1DS telemetry
HTTP worker can lock an already-destroyed mutex during interpreter teardown and abort the process
(`Abort trap: 6`, exit 134) *after* every test has passed:

    recursive_mutex::lock() -> __throw_system_error
    Microsoft::Applications::Events::DebugEventSource::DispatchEvent
    Microsoft::Applications::Events::HttpResponseDecoder::handleDecode
    Microsoft::Applications::Events::HttpClientManager::onHttpResponse

The throw comes from a background thread and is never caught, so it takes the whole run down and
reds CI on an unrelated change. No MLX frame appears in that stack and it reproduces on builds
predating this EP's float64 work, so it is ONNX Runtime's, not ours — but disabling telemetry
removes the thread that does it.

`setdefault`, not assignment: an explicit `ORT_DISABLE_TELEMETRY=0` in the environment still wins,
so this can be turned off deliberately without editing the file.
"""

from __future__ import annotations

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
