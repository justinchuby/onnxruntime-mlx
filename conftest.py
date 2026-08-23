"""Repo-wide pytest configuration.

Exists for one reason: **ONNX Runtime telemetry is disabled for local test runs, and left on in
CI.**

CI keeps telemetry so ONNX Runtime's maintainers see the runtime is being used — that signal is
worth something to a project we depend on, and a CI machine is not anybody's personal machine. What
we don't want is a developer's laptop phoning home every time they run the suite. So the rule is
scoped by where it runs rather than applied everywhere:

    local -> ORT_DISABLE_TELEMETRY=1
    CI    -> left unset, i.e. ORT's own default (telemetry on)

`ORT_DISABLE_TELEMETRY` is read by ONNX Runtime when its native library is loaded, so it has to be
in the environment before anything imports `onnxruntime`. A fixture — even a session-scoped autouse
one — runs too late, because the suite conftests import `onnxruntime` at module scope. The rootdir
conftest is imported before those, which makes this the only place the assignment is reliably early
enough.

`setdefault`, not assignment: an explicit `ORT_DISABLE_TELEMETRY` in the environment wins in either
direction, so a developer can opt back in and a CI job can opt out without editing this file.

Known consequence of leaving telemetry on in CI
-----------------------------------------------
ORT's 1DS telemetry HTTP worker can lock an already-destroyed mutex during interpreter teardown and
abort the process (`Abort trap: 6`, exit 134) *after* every test has passed:

    recursive_mutex::lock() -> __throw_system_error
    Microsoft::Applications::Events::DebugEventSource::DispatchEvent
    Microsoft::Applications::Events::HttpResponseDecoder::handleDecode
    Microsoft::Applications::Events::HttpClientManager::onHttpResponse

The throw comes from a background thread and is never caught, so it takes the whole run down. No MLX
frame appears in that stack and it reproduces on builds predating this EP's float64 work, so it is
ONNX Runtime's, not ours. Disabling telemetry removes the thread that does it, which is why local
runs are quiet; CI accepts the risk in exchange for the usage signal. If a CI job flakes on exit 134
with a green test summary, that is this and not the change under test — set
`ORT_DISABLE_TELEMETRY: "1"` on the affected workflow rather than debugging the PR.
"""

from __future__ import annotations

import os

# `CI` is the de-facto standard flag; GitHub Actions sets both CI=true and GITHUB_ACTIONS=true.
IN_CI = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))

if not IN_CI:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
