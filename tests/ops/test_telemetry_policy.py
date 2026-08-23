"""Guards the repo-wide rule that ONNX Runtime telemetry is disabled.

The rule is enforced in several places that cannot share code — the rootdir `conftest.py`, four CI
workflows, `bench/bench.py`, and `tests/conformance/run_conformance.sh` — because each is a separate
entry point that reaches ONNX Runtime without passing through the others. Config duplicated across
files drifts, so the parts that can be checked are checked here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = ("ci.yml", "bench.yml", "conformance.yml", "publish.yml")


def test_telemetry_is_disabled_in_this_process():
    """The rootdir conftest must have set this before `onnxruntime` was imported.

    Asserted on the environment rather than on ORT state because ORT exposes no way to read the
    setting back; the environment is what it consults at library load.
    """
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1", (
        "ORT_DISABLE_TELEMETRY is not set for the test process. It has to be assigned in the "
        "rootdir conftest.py, before anything imports onnxruntime — a fixture runs too late."
    )


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_workflow_disables_telemetry(workflow):
    """Every workflow sets it at workflow level, so all jobs and steps inherit it.

    Job-level or step-level would leave the ORT-invoking steps that don't run pytest uncovered.
    """
    yaml = pytest.importorskip("yaml")
    path = _REPO_ROOT / ".github" / "workflows" / workflow
    config = yaml.safe_load(path.read_text())
    assert config.get("env", {}).get("ORT_DISABLE_TELEMETRY") == "1", (
        f"{workflow} does not set ORT_DISABLE_TELEMETRY at workflow level"
    )


def test_non_pytest_entry_points_disable_telemetry():
    """`bench/bench.py` and the conformance runner reach ORT without loading the rootdir conftest.

    bench.py is run directly; run_conformance.sh drives onnx-tests' pytest from *its* checkout, so
    this repo's conftest is never collected. Both must set the variable themselves.
    """
    bench = (_REPO_ROOT / "bench" / "bench.py").read_text()
    assert 'os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")' in bench, (
        "bench/bench.py must disable telemetry itself"
    )
    # It is only effective before ORT's native library loads.
    assert bench.index("ORT_DISABLE_TELEMETRY") < bench.index("import onnxruntime"), (
        "bench/bench.py sets ORT_DISABLE_TELEMETRY after importing onnxruntime, which is too late"
    )

    conformance = (_REPO_ROOT / "tests" / "conformance" / "run_conformance.sh").read_text()
    assert "ORT_DISABLE_TELEMETRY" in conformance, (
        "run_conformance.sh must export ORT_DISABLE_TELEMETRY for the onnx-tests subprocesses"
    )
