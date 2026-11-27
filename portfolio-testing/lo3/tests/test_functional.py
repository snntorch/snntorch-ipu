"""Check neuron dynamics, surrogate gradients and encoders against independent expectations."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import snntorch as snn
from snntorch import surrogate, spikegen

HERE = Path(__file__).resolve().parents[1]
DESIGN = json.loads((HERE / "design.json").read_text())
PAIRS = json.loads((HERE / "pairwise.json").read_text())["cases"]


def close(observed, expected, dtype):
    torch.testing.assert_close(
        observed,
        torch.as_tensor(expected, dtype=observed.dtype),
        check_dtype=False,
        **DESIGN["numerics"][dtype],
    )


def reference(currents, beta, threshold, reset):
    """Use the previous membrane for reset and the updated membrane for firing."""
    mem = np.zeros_like(currents[0], dtype=np.float64)
    trace, spikes = [], []
    for current in currents:
        fired = mem > threshold
        nxt = np.clip(beta, 0, 1) * mem + current
        if reset == "subtract":
            nxt = nxt - fired * threshold
        if reset == "zero":
            nxt = np.where(fired, 0, nxt)
        mem = nxt
        trace.append(mem.copy())
        spikes.append((mem > threshold).astype(float))
    return np.array(spikes), np.array(trace)


@pytest.mark.parametrize("case", PAIRS, ids=lambda c: c["id"])
def test_R1_pairwise(case, evidence):
    dtype = getattr(torch, case["dtype"])
    beta = [0.0, 0.5, 1.0] if case["parameter"] == "vector" else 0.5
    threshold = [0.5, 1.0, 1.5] if case["parameter"] == "vector" else 1.0
    neuron = snn.Leaky(
        beta=torch.tensor(beta, dtype=dtype),
        threshold=torch.tensor(threshold, dtype=dtype),
        reset_mechanism=case["reset"],
        init_hidden=case["hidden"],
        output=True,
    )
    anchor = [[0.5, 1.0, 1.5], [1.25, 0.5, -0.25], [0, 2, 1], [-1, 0, 0]]
    currents = torch.tensor([anchor[i % 4] for i in range(case["steps"])], dtype=dtype)
    currents = currents[:, None, :].repeat(1, case["batch"], 1)
    expected_s, expected_m = reference(
        currents.numpy(), np.array(beta), np.array(threshold), case["reset"]
    )
    mem = torch.zeros_like(currents[0])
    actual_s, actual_m = [], []
    for current in currents:
        if case["hidden"]:
            spk, mem = neuron(current)
        else:
            spk, mem = neuron(current, mem)
        actual_s.append(spk.detach())
        actual_m.append(mem.detach())
    evidence(
        requirement="R1",
        technique="pairwise",
        config=case,
        input=currents.tolist(),
        expected_mem=expected_m.tolist(),
        observed_mem=torch.stack(actual_m).tolist(),
        expected_spikes=expected_s.tolist(),
        observed_spikes=torch.stack(actual_s).tolist(),
    )
    assert torch.equal(
        torch.stack(actual_s), torch.tensor(expected_s, dtype=torch.float32)
    )
    close(torch.stack(actual_m), expected_m, case["dtype"])


@pytest.mark.parametrize(
    "reset,mem,spikes",
    [
        ("subtract", [1, 1.5, 0.75, 1.375], [0, 1, 0, 1]),
        ("zero", [1, 1.5, 0, 1], [0, 1, 0, 0]),
        ("none", [1, 1.5, 1.75, 1.875], [0, 1, 1, 1]),
    ],
)
def test_R1_hand_anchor(reset, mem, spikes, evidence):
    n = snn.Leaky(0.5, reset_mechanism=reset)
    u = torch.zeros(1)
    got = [n(torch.ones(1), u)]
    for _ in range(3):
        got.append(n(torch.ones(1), got[-1][1]))
    evidence(
        requirement="R1", expected_mem=mem, observed_mem=[x[1].item() for x in got]
    )
    assert [x[1].item() for x in got] == mem
    assert [x[0].item() for x in got] == spikes


@pytest.mark.parametrize("beta", [-1.0, 0.0, 1.0, 2.0])
def test_R1_beta_clipping(beta):
    n = snn.Leaky(beta, reset_mechanism="none")
    _, u = n(torch.zeros(3), torch.ones(3))
    close(u, [min(1, max(0, beta))] * 3, "float32")


@pytest.mark.parametrize("hidden", [False, True])
def test_R1_threshold_neighbours(hidden):
    t = torch.tensor(1.0)
    values = torch.stack(
        [
            torch.nextafter(t, torch.tensor(0.0)),
            t,
            torch.nextafter(t, torch.tensor(2.0)),
        ]
    )
    n = snn.Leaky(0.5, init_hidden=hidden, output=True)
    spk, _ = n(values) if hidden else n(values, torch.zeros_like(values))
    assert spk.tolist() == [0, 0, 1]


@pytest.mark.parametrize("hidden", [False, True])
def test_R1_synaptic_equations(hidden, evidence):
    n = snn.Synaptic(
        alpha=0.5, beta=0.5, init_hidden=hidden, output=True, reset_mechanism="none"
    )
    syn = mem = torch.zeros(1)
    expected = [(1, 1), (1.5, 2), (1.75, 2.75)]
    got = []
    for _ in expected:
        result = n(torch.ones(1)) if hidden else n(torch.ones(1), syn, mem)
        _, syn, mem = result
        got.append((syn.item(), mem.item()))
    evidence(requirement="R1", expected=expected, observed=got)
    assert got == expected


def test_R1_rejection():
    with pytest.raises(ValueError, match="reset_mechanism"):
        snn.Leaky(0.5, reset_mechanism="invalid")
    n = snn.Leaky(0.5, init_hidden=True)
    with pytest.raises(TypeError, match="1 input"):
        n(torch.ones(1), torch.zeros(1))


SURROGATES = [
    "ste",
    "triangular",
    "fast",
    "sigmoid",
    "escape",
    "lso",
    "sfs",
    "sso_zero",
]


@pytest.mark.parametrize("name", SURROGATES)
@pytest.mark.parametrize("parameter", [0.5, 2.0])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_R2_surrogate(name, parameter, dtype, evidence):
    vals = [-2, -0.5, -1e-6, 0, 1e-6, 0.5, 1, 1.5, 2]
    x = torch.tensor(
        [vals, vals[::-1]], dtype=getattr(torch, dtype), requires_grad=True
    )
    if name == "ste":
        fn, deriv = surrogate.straight_through_estimator(), lambda v: 1
    elif name == "triangular":
        fn, deriv = (
            lambda x: surrogate.Triangular.apply(x, parameter),
            lambda v: parameter if v < 0 else -parameter,
        )
    elif name == "fast":
        fn, deriv = (
            surrogate.fast_sigmoid(parameter),
            lambda v: 1 / (1 + parameter * abs(v)) ** 2,
        )
    elif name == "sigmoid":
        fn, deriv = (
            surrogate.sigmoid(parameter),
            lambda v: (
                parameter
                * math.exp(-parameter * v)
                / (1 + math.exp(-parameter * v)) ** 2
            ),
        )
    elif name == "escape":
        fn, deriv = (
            surrogate.spike_rate_escape(beta=parameter, slope=3),
            lambda v: 3 * math.exp(-parameter * abs(v - 1)),
        )
    elif name == "lso":
        fn, deriv = surrogate.LSO(parameter), lambda v: 1 if v > 0 else parameter
    elif name == "sfs":
        fn, deriv = (
            surrogate.SFS(parameter, B=0.5),
            lambda v: 1 / (1 + parameter * abs(v)) ** 2 if v > 0.5 else 0,
        )
    else:
        fn, deriv = (
            surrogate.SSO(mean=parameter, variance=0),
            lambda v: 1 if v > 0 else 0,
        )
    y = fn(x)
    weights = torch.tensor([[1, -1, 2, 0.5, 1, -2, 1, 2, 1]] * 2, dtype=x.dtype)
    (y * weights).sum().backward()
    expected = [[deriv(v) for v in row] for row in x.detach().tolist()]
    expected = torch.tensor(expected, dtype=x.dtype) * weights
    evidence(
        requirement="R2",
        name=name,
        parameter=parameter,
        dtype=dtype,
        input=x.detach().tolist(),
        expected_gradient=expected.tolist(),
        observed_gradient=x.grad.tolist(),
    )
    assert torch.equal(y, (x > 0).float())
    close(x.grad, expected, dtype)


def test_R2_triangular_public_wrapper():
    x = torch.tensor(0.25, requires_grad=True)
    surrogate.triangular()(x).backward()
    assert x.grad.item() == -1


def test_R2_stochastic_support(evidence):
    torch.manual_seed(71)
    x = torch.tensor([-1.0, 0, 1.0] * 100, requires_grad=True)
    surrogate.SSO(mean=0.25, variance=0.4)(x).sum().backward()
    negative = x.grad[x <= 0]
    evidence(
        requirement="R2",
        seed=71,
        observed_min=negative.min().item(),
        observed_max=negative.max().item(),
        bounds=[-0.1, 0.3],
    )
    assert torch.all(negative >= -0.1) and torch.all(negative < 0.3)
    assert torch.all(x.grad[x > 0] == 1)


def test_R2_autograd_integration(evidence):
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.5)
    neuron = snn.Leaky(
        beta=0.5,
        spike_grad=surrogate.fast_sigmoid(2),
        learn_beta=True,
        learn_threshold=True,
    )
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    spk, mem = neuron(layer(x), torch.tensor([[0.25]]))
    spk.sum().backward()
    d = 1 / (1 + 2 * 0.625) ** 2
    close(x.grad, [[0.5 * d, 0.5 * d]], "float32")
    close(layer.weight.grad, [[d, 2 * d]], "float32")
    close(neuron.beta.grad, 0.25 * d, "float32")
    close(neuron.threshold.grad, -d, "float32")
    evidence(requirement="R2", gradient=x.grad.tolist(), expected=[[0.5 * d, 0.5 * d]])


@pytest.mark.parametrize(
    "case",
    json.loads((HERE / "rate-intervals.json").read_text())["cases"],
    ids=lambda c: f"p{c['p']}-seed{c['seed']}",
)
def test_R3_rate_statistical(case, evidence):
    torch.manual_seed(case["seed"])
    spikes = spikegen.rate(torch.tensor([[case["p"]]]), num_steps=case["n"])
    count = int(spikes.sum())
    evidence(requirement="R3", **case, observed=count)
    assert case["low"] <= count <= case["high"]


def test_R3_rate_exact():
    got = spikegen.rate(torch.tensor([[0.0, 1.0]]), num_steps=8, first_spike_time=2)
    assert got.shape == (8, 1, 2)
    assert not got[:2].any() and not got[:, :, 0].any() and got[2:, :, 1].all()
    got = spikegen.rate(
        torch.tensor([[0.0, 1.0]]), time_var_input=True, first_spike_time=2
    )
    assert got.tolist() == [[0, 0], [0, 0], [0, 1]]
    assert spikegen.rate(torch.tensor([[-1.0, 2.0]]), num_steps=1).tolist() == [
        [[0, 1]]
    ]


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("first", [0, 2])
def test_R3_latency(linear, clip, first, evidence):
    values = [0.0, 0.1, 0.1001, 0.5, 1.0]
    steps, tau, threshold = 20, 1, 0.1
    # Use the same float32 values that the encoder actually receives.
    data = torch.tensor([values])
    values = data[0].tolist()
    times = [
        min(tau * (1 - v), tau * (1 - threshold)) + first
        if linear
        else tau
        * math.log(max(v, threshold + 1e-7) / (max(v, threshold + 1e-7) - threshold))
        + first
        for v in values
    ]
    expected = torch.zeros(steps, 1, len(values))
    for col, (v, t) in enumerate(zip(values, times)):
        if not clip or v >= threshold:
            expected[round(t), 0, col] = 1
    got = spikegen.latency(
        data,
        num_steps=steps,
        threshold=threshold,
        tau=tau,
        linear=linear,
        clip=clip,
        first_spike_time=first,
    )
    evidence(
        requirement="R3",
        input=values,
        expected_times=times,
        observed_nonzero=got.nonzero().tolist(),
    )
    assert torch.equal(got, expected)


def test_R3_latency_normalized():
    got = spikegen.latency(
        torch.tensor([0.02, 0.5, 1.0]), num_steps=5, normalize=True, linear=True
    )
    assert got.nonzero().tolist() == [[0, 2], [2, 1], [4, 0]]


def test_R3_one_hot():
    labels = torch.tensor([2, 0, 2, 1, 0])
    got = spikegen.to_one_hot(labels, 3)
    expected = torch.tensor([[int(label == c) for c in range(3)] for label in labels])
    assert torch.equal(got, expected)
    assert got.shape == (5, 3)


@pytest.mark.parametrize(
    "kind",
    [
        "rate_negative_steps",
        "rate_negative_first",
        "rate_conflict",
        "rate_first_oob",
        "latency_low",
        "latency_high",
        "latency_tau",
        "latency_threshold0",
        "latency_threshold1",
        "latency_first",
        "onehot_high",
        "onehot_negative",
        "onehot_shape",
    ],
)
def test_R3_invalid(kind):
    calls = {
        "rate_negative_steps": lambda: spikegen.rate(torch.ones(1), num_steps=-1),
        "rate_negative_first": lambda: spikegen.rate(
            torch.ones(1), num_steps=2, first_spike_time=-1
        ),
        "rate_conflict": lambda: spikegen.rate(
            torch.ones(1), num_steps=2, time_var_input=True
        ),
        "rate_first_oob": lambda: spikegen.rate(
            torch.ones(1), num_steps=2, first_spike_time=2
        ),
        "latency_low": lambda: spikegen.latency(torch.tensor([-0.01]), num_steps=5),
        "latency_high": lambda: spikegen.latency(torch.tensor([1.01]), num_steps=5),
        "latency_tau": lambda: spikegen.latency(torch.ones(1), num_steps=5, tau=0),
        "latency_threshold0": lambda: spikegen.latency(
            torch.ones(1), num_steps=5, threshold=0
        ),
        "latency_threshold1": lambda: spikegen.latency(
            torch.ones(1), num_steps=5, threshold=1
        ),
        "latency_first": lambda: spikegen.latency(
            torch.ones(1), num_steps=5, first_spike_time=-1
        ),
        "onehot_high": lambda: spikegen.to_one_hot(torch.tensor([3]), 3),
        "onehot_negative": lambda: spikegen.to_one_hot(torch.tensor([-1]), 3),
        "onehot_shape": lambda: spikegen.to_one_hot(torch.tensor([[1]]), 3),
    }
    with pytest.raises(Exception):
        calls[kind]()
