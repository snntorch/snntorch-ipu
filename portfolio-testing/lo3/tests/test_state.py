"""Check that resets start a fresh sequence without changing a peer model."""

import itertools
import json
from pathlib import Path

import pytest
import torch
import snntorch as snn
from snntorch import utils

HERE = Path(__file__).resolve().parents[1]
CASES = json.loads((HERE / "model-cases.json").read_text())


def factory(name, reset, depth, dtype):
    cls = getattr(snn, name)
    kw = dict(beta=0.5, reset_mechanism=reset, init_hidden=True)
    if name == "Synaptic":
        kw["alpha"] = 0.25
    return torch.nn.Sequential(*[cls(**kw) for _ in range(depth)]).to(dtype=dtype)


def snapshot(net):
    return {
        f"{i}.{key}": getattr(layer, key).detach().clone()
        for i, layer in enumerate(net)
        for key in ("mem", "syn")
        if hasattr(layer, key) and not hasattr(getattr(layer, key), "init_flag")
    }


def process(net, inputs):
    return torch.stack([net(x).detach() for x in inputs]), snapshot(net)


def equal_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def serial(state):
    return {k: v.tolist() for k, v in state.items()}


def tour():
    # Set up each of the four states before trying its four actions.
    for source in itertools.product([0, 1], repeat=2):
        for action in ["processA", "processB", "resetA", "resetB"]:
            yield source, action


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_R4_model_sequence(case, evidence):
    dtype = getattr(torch, case["dtype"])
    gen = torch.Generator().manual_seed(case["seed"])
    seq_a = (
        torch.rand(case["steps"], case["batch"], 3, generator=gen, dtype=dtype) * 1.5
    )
    seq_b = (
        torch.rand(case["steps"] + 1, case["batch"], 3, generator=gen, dtype=dtype)
        * 1.5
    )
    a = factory(case["model"], case["reset"], case["depth"], dtype)
    b = factory(case["peer"], case["reset"], case["depth"], dtype)
    fresh = factory(case["model"], case["reset"], case["depth"], dtype)
    process(a, seq_a)
    utils.reset(a)
    expected_out, expected_state = process(fresh, seq_b)
    got_out, got_state = process(a, seq_b)
    evidence(
        requirement="R4",
        case=case,
        seq_a=seq_a.tolist(),
        seq_b=seq_b.tolist(),
        fresh_expected=serial(expected_state),
        reset_observed=serial(got_state),
    )
    assert torch.equal(got_out, expected_out)
    equal_state(got_state, expected_state)
    process(b, seq_b)
    b_expected = factory(case["peer"], case["reset"], case["depth"], dtype)
    process(b_expected, seq_b)
    before = snapshot(b)
    utils.reset(a)
    evidence(
        requirement="R4",
        case_id=case["id"],
        peer_before=serial(before),
        peer_after=serial(snapshot(b)),
    )
    equal_state(snapshot(b), before)
    out_b, state_b = process(b, seq_a)
    ref_b, ref_state_b = process(b_expected, seq_a)
    assert torch.equal(out_b, ref_b)
    equal_state(state_b, ref_state_b)
    states, edges, log = set(), set(), []
    state = [1, 1]

    def act(action):
        source = tuple(state)
        target_index = 0 if action.endswith("A") else 1
        target, peer = (a, b) if target_index == 0 else (b, a)
        peer_before = snapshot(peer)
        if action.startswith("reset"):
            utils.reset(target)
            state[target_index] = 0
            for value in snapshot(target).values():
                assert not value.any()
        else:
            process(target, seq_a)
            state[target_index] = 1
        equal_state(snapshot(peer), peer_before)
        states.update([source, tuple(state)])
        edges.add((source, action, tuple(state)))
        log.append([list(source), action, list(state)])

    for source, action in tour():
        act("processA" if source[0] else "resetA")
        act("processB" if source[1] else "resetB")
        act(action)
    evidence(
        requirement="R4",
        case_id=case["id"],
        states=[list(s) for s in sorted(states)],
        transitions=log,
        unique_transitions=len(edges),
        final_a=serial(snapshot(a)),
        final_b=serial(snapshot(b)),
    )
    assert len(states) == 4 and len(edges) == 16


@pytest.mark.parametrize("peer", ["Leaky", "Synaptic"])
def test_R4_class_scope(peer):
    a = factory("Leaky", "subtract", 1, torch.float32)
    b = factory(peer, "subtract", 1, torch.float32)
    x = torch.ones(1, 1, 3) * 0.75
    process(a, x)
    process(b, x)
    before = snapshot(b)
    snn.Leaky.reset_hidden()
    assert hasattr(a[0].mem, "init_flag")
    if peer == "Leaky":
        assert hasattr(b[0].mem, "init_flag")
    else:
        equal_state(snapshot(b), before)


def test_R4_nested_and_global_scope():
    a = factory("Leaky", "subtract", 1, torch.float32)
    b = factory("Synaptic", "subtract", 1, torch.float32)
    process(a, torch.ones(1, 1, 3))
    process(b, torch.ones(1, 1, 3))
    root = torch.nn.ModuleList([a, b])
    utils.reset(root)
    for net in (a, b):
        for value in snapshot(net).values():
            assert not value.any()


def test_R4_reset_zero_one_many():
    for count in [0, 1, 3]:
        net = torch.nn.Sequential(
            *[snn.Leaky(0.5, init_hidden=True) for _ in range(count)]
        )
        if count:
            net(torch.ones(1, 2))
        utils.reset(net)
        utils.reset(net)
        for value in snapshot(net).values():
            assert not value.any()
