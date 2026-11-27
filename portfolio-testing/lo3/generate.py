"""Generate the case lists and rate intervals before running the tests."""

import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate():
    factors = {
        "reset": ["subtract", "zero", "none"],
        "parameter": ["scalar", "vector"],
        "steps": [1, 2, 8],
        "batch": [1, 3],
        "dtype": ["float32", "float64"],
        "hidden": [False, True],
        "device": ["cpu"],
    }
    candidates = list(itertools.product(*factors.values()))

    # Every combination in this CPU settings table is allowed.
    def pairs(row):
        return {
            (i, row[i], j, row[j])
            for i in range(len(row))
            for j in range(i + 1, len(row))
        }

    remaining = set().union(*(pairs(c) for c in candidates))
    chosen = []
    while remaining:
        row = max(candidates, key=lambda c: len(pairs(c) & remaining))
        chosen.append(row)
        remaining -= pairs(row)
    dump(
        "pairwise.json",
        {
            "factors": factors,
            "feasible_configurations": len(candidates),
            "cases": [
                dict(id=f"PW-{i:03}", **dict(zip(factors, row)))
                for i, row in enumerate(chosen)
            ],
        },
    )
    combos = list(
        itertools.product(
            ["Leaky", "Synaptic"],
            ["Leaky", "Synaptic"],
            factors["reset"],
            factors["steps"],
            factors["batch"],
            factors["dtype"],
            [1, 2],
            range(5),
        )
    )
    random.Random(20261102).shuffle(combos)
    cases = [
        dict(
            id=f"MB-{i:04}",
            model=a,
            peer=b,
            reset=r,
            steps=t,
            batch=n,
            dtype=d,
            depth=z,
            seed=10000 + s + 5 * i,
        )
        for i, (a, b, r, t, n, d, z, s) in enumerate(combos[:1000])
    ]
    dump("model-cases.json", cases)
    from scipy.stats import binom

    design = json.loads((HERE / "design.json").read_text())
    rate = design["rate"]
    alpha = rate["family_alpha"] / (len(rate["probabilities"]) * len(rate["seeds"]))
    dump(
        "rate-intervals.json",
        {
            "per_count_alpha": alpha,
            "cases": [
                dict(
                    p=p,
                    seed=s,
                    n=rate["n"],
                    low=int(binom.ppf(alpha / 2, rate["n"], p)),
                    high=int(binom.ppf(1 - alpha / 2, rate["n"], p)),
                )
                for p in rate["probabilities"]
                for s in rate["seeds"]
            ],
        },
    )


if __name__ == "__main__":
    generate()
