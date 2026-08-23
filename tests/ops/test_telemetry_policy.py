"""Guards the repo-wide telemetry rule: **off on developer machines, on in CI.**

CI keeps ONNX Runtime's telemetry so its maintainers see the runtime being used; a laptop should not
phone home on every `pytest` run. The rule is therefore conditional on `CI`/`GITHUB_ACTIONS`, and it
is enforced in three places that cannot share code — the rootdir `conftest.py`, `bench/bench.py`,
and `tests/conformance/run_conformance.sh` — because each is a separate entry point that reaches
ONNX Runtime without passing through the others.

The workflows are checked for the *absence* of a force-disable, which is the direction that would
silently cost the usage signal if someone reintroduced it out of habit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = ("ci.yml", "bench.yml", "conformance.yml", "publish.yml")
_IN_CI = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def test_telemetry_setting_matches_the_environment():
    """The rootdir conftest must have decided this before `onnxruntime` was imported.

    Asserted on the environment rather than on ORT state because ORT exposes no way to read the
    setting back; the environment is what it consults at library load.
    """
    setting = os.environ.get("ORT_DISABLE_TELEMETRY")
    if _IN_CI:
        assert setting != "1", (
            "telemetry is disabled in CI, which drops the usage signal ONNX Runtime gets from us. "
            "The rootdir conftest should only set ORT_DISABLE_TELEMETRY when not in CI."
        )
    else:
        assert setting == "1", (
            "ORT_DISABLE_TELEMETRY is not set for this local test process. It has to be assigned "
            "in the rootdir conftest.py, before anything imports onnxruntime — a fixture is too "
            "late."
        )


def test_local_runs_disable_telemetry_even_when_ci_vars_are_absent():
    """Directly exercise the conftest's decision in both environments.

    The in-process test above can only observe whichever branch this run happens to take, so the
    other branch would rot unnoticed. Importing the conftest in a subprocess with a controlled
    environment covers both.
    """
    conftest = _REPO_ROOT / "conftest.py"
    program = (
        "import runpy, os;"
        "ns = runpy.run_path(r'%s');"
        "print(os.environ.get('ORT_DISABLE_TELEMETRY', '<unset>'))" % conftest
    )

    local_env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
    local_env.pop("ORT_DISABLE_TELEMETRY", None)
    local = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=local_env, check=True
    )
    assert local.stdout.strip() == "1", (
        f"a local run must disable telemetry, got {local.stdout.strip()!r}"
    )

    ci_env = dict(local_env, CI="true")
    ci = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=ci_env, check=True
    )
    assert ci.stdout.strip() == "<unset>", (
        f"a CI run must leave telemetry at ORT's default, got {ci.stdout.strip()!r}"
    )

    # An explicit value wins in either direction, so this can be overridden without editing files.
    override = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=dict(ci_env, ORT_DISABLE_TELEMETRY="1"),
        check=True,
    )
    assert override.stdout.strip() == "1", "an explicit ORT_DISABLE_TELEMETRY must be respected"


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_workflow_does_not_force_telemetry_off(workflow):
    """CI is where we deliberately leave telemetry on, so no workflow may globally disable it.

    A job that genuinely needs it off can still set it at job or step level; this only guards the
    workflow-level blanket setting that used to be here.
    """
    yaml = pytest.importorskip("yaml")
    path = _REPO_ROOT / ".github" / "workflows" / workflow
    config = yaml.safe_load(path.read_text())
    workflow_env = config.get("env") or {}
    assert workflow_env.get("ORT_DISABLE_TELEMETRY") != "1", (
        f"{workflow} disables ONNX Runtime telemetry for the whole workflow, which removes the "
        "usage signal CI is meant to provide. Scope it to a job or step if some specific step "
        "needs it."
    )


def test_non_pytest_entry_points_apply_the_same_rule():
    """`bench/bench.py` and the conformance runner reach ORT without loading the rootdir conftest.

    bench.py is run directly; run_conformance.sh drives onnx-tests' pytest from *its* checkout, so
    this repo's conftest is never collected. Both must make the same local-vs-CI decision, and both
    must make it before ORT's native library loads.
    """
    bench = (_REPO_ROOT / "bench" / "bench.py").read_text()
    assert 'os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")' in bench, (
        "bench/bench.py must disable telemetry for local runs"
    )
    assert 'GITHUB_ACTIONS' in bench and 'CI' in bench, (
        "bench/bench.py must leave telemetry on in CI, matching the rootdir conftest"
    )
    assert bench.index("ORT_DISABLE_TELEMETRY") < bench.index("import onnxruntime"), (
        "bench/bench.py sets ORT_DISABLE_TELEMETRY after importing onnxruntime, which is too late"
    )

    conformance = (_REPO_ROOT / "tests" / "conformance" / "run_conformance.sh").read_text()
    assert "ORT_DISABLE_TELEMETRY" in conformance, (
        "run_conformance.sh must set ORT_DISABLE_TELEMETRY for the onnx-tests subprocesses"
    )
    assert "GITHUB_ACTIONS" in conformance, (
        "run_conformance.sh must leave telemetry on in CI, matching the rootdir conftest"
    )
