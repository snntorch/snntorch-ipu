"""Isolate the neuron registry for each test and save case observations."""

import json
import os
from pathlib import Path

import pytest
import torch
import snntorch as snn

HERE = Path(__file__).resolve().parents[1]
RECORDS = []


@pytest.fixture(autouse=True)
def isolate_registry():
    previous = snn.SpikingNeuron.instances
    snn.SpikingNeuron.instances = []
    yield
    snn.SpikingNeuron.instances = previous


@pytest.fixture
def evidence(request):
    records = []

    def save(**values):
        records.append(values)
        request.node.user_properties[:] = [("evidence", json.dumps(records))]

    return save


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or (report.when == "setup" and report.failed):
        RECORDS.append(
            dict(
                id=report.nodeid,
                outcome=report.outcome,
                duration=report.duration,
                details=dict(report.user_properties),
                failure=str(report.longrepr) if report.failed else None,
            )
        )


def pytest_sessionfinish(session, exitstatus):
    dest = os.environ.get("LO3_RESULTS")
    if dest:
        Path(dest).write_text(json.dumps(RECORDS, indent=2), encoding="utf-8")
