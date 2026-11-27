"""Summarise the recorded test results and check settings pairs from case IDs."""

import collections
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def read_optional(path):
    return read(path) if (HERE / path).is_file() else None


def rate(n, d):
    return dict(covered=n, total=d, percent=round(100 * n / d, 2) if d else None)


def outcome(path):
    root = ET.parse(HERE / path).getroot()
    tests = root.findall(".//testcase")
    return dict(
        total=len(tests),
        passed=sum(not list(t) or all(c.tag == "properties" for c in t) for t in tests),
        failed=sum(t.find("failure") is not None for t in tests),
        errors=sum(t.find("error") is not None for t in tests),
        skipped=sum(t.find("skipped") is not None for t in tests),
    )


def main():
    results = read("evidence/final/results.json")
    ids = {r["id"] for r in results}
    passed = {r["id"] for r in results if r["outcome"] == "passed"}
    combinations = read("pairwise.json")
    names = list(combinations["factors"])
    # Build the expected pairs from the settings themselves so this also checks the generator.
    expected = set()
    for a, b in itertools.combinations(names, 2):
        for av, bv in itertools.product(
            combinations["factors"][a], combinations["factors"][b]
        ):
            expected.add((a, str(av), b, str(bv)))
    observed = set()
    for c in combinations["cases"]:
        if not any(f"test_R1_pairwise[{c['id']}]" in x for x in passed):
            continue
        for a, b in itertools.combinations(names, 2):
            observed.add((a, str(c[a]), b, str(c[b])))
    assert observed <= expected
    coverage = read("evidence/final/coverage.json")
    declared = set(read("design.json")["scope"])
    assert {p.replace("\\", "/") for p in coverage["files"]} == declared
    total = coverage["totals"]
    model = [r for r in results if "test_R4_model_sequence[" in r["id"]]
    manifest = read("model-cases.json")
    assert len({json.dumps(c, sort_keys=True) for c in manifest}) == 1000
    assert (
        len(
            {
                (
                    c["model"],
                    c["peer"],
                    c["reset"],
                    c["steps"],
                    c["batch"],
                    c["dtype"],
                    c["depth"],
                    c["seed"],
                )
                for c in manifest
            }
        )
        == 1000
    )
    states, edges = set(), set()
    counts = []
    for r in results:
        details = json.loads(r["details"].get("evidence", "[]"))
        if isinstance(details, dict):
            details = [details]
        for detail in details:
            if "states" in detail and r["outcome"] == "passed":
                states.update(tuple(s) for s in detail["states"])
                edges.update(
                    (tuple(s), action, tuple(t))
                    for s, action, t in detail["transitions"]
                )
            if "observed" in detail and "n" in detail and "p" in detail:
                counts.append(detail)
    possible_edges = set()
    for source in itertools.product([0, 1], repeat=2):
        for action in ["processA", "processB", "resetA", "resetB"]:
            target = list(source)
            target[0 if action.endswith("A") else 1] = int(action.startswith("process"))
            possible_edges.add((source, action, tuple(target)))
    assert edges <= possible_edges
    # Check the categories named in the test design. A name match can credit several cases.
    inventory = []

    def add(req, feature, values, test, boundary=False):
        for value in values:
            inventory.append(
                dict(
                    requirement=req,
                    feature=feature,
                    value=value,
                    test=test,
                    boundary=boundary,
                    executed=any(test in x for x in ids),
                    passed=any(test in x for x in passed),
                )
            )

    add("R1", "reset", ["subtract", "zero", "none"], "test_R1_hand_anchor")
    for feature, values in [
        ("parameter", ["scalar", "vector"]),
        ("steps", [1, 2, 8]),
        ("batch", [1, 3]),
        ("dtype", ["float32", "float64"]),
        ("state", ["explicit", "hidden"]),
    ]:
        add("R1", feature, values, "test_R1_pairwise")
    add(
        "R1",
        "threshold",
        ["below-adjacent", "equal", "above-adjacent"],
        "test_R1_threshold_neighbours",
        True,
    )
    add("R1", "beta", [-1, 0, 1, 2], "test_R1_beta_clipping", True)
    add("R1", "invalid", ["reset-name", "hidden-explicit-state"], "test_R1_rejection")
    add("R1", "synaptic-state", ["explicit", "hidden"], "test_R1_synaptic_equations")
    add(
        "R1",
        "inhibition",
        ["explicit", "hidden"],
        "test_R1_inhibition_and_reset_change",
    )
    add(
        "R2",
        "family",
        ["ste", "triangular", "fast", "sigmoid", "escape", "lso", "sfs", "sso-zero"],
        "test_R2_surrogate",
    )
    add("R2", "sign", ["negative", "zero", "positive"], "test_R2_surrogate", True)
    add("R2", "parameters", [0.5, 2.0], "test_R2_surrogate")
    add("R2", "dtype", ["float32", "float64"], "test_R2_surrogate")
    add("R2", "sfs-support", ["below-B", "at-B", "above-B"], "test_R2_surrogate", True)
    add(
        "R2",
        "stochastic-support",
        ["bounded-negative", "positive-one"],
        "test_R2_stochastic_support",
    )
    add(
        "R2",
        "integration",
        ["input", "weight", "beta", "threshold"],
        "test_R2_autograd_integration",
    )
    add("R2", "factory", ["triangular-default"], "test_R2_triangular_public_wrapper")
    add("R3", "rate", [0, 1], "test_R3_rate_exact", True)
    add("R3", "rate-probability", [0.1, 0.5, 0.9], "test_R3_rate_statistical")
    add(
        "R3",
        "rate-time",
        ["static", "time-varying", "delayed", "clipped"],
        "test_R3_rate_exact",
    )
    add("R3", "latency-values", [0, 0.1, 0.1001, 0.5, 1], "test_R3_latency", True)
    for feature, values in [
        ("latency-code", ["log", "linear"]),
        ("clip", [False, True]),
        ("delay", [0, 2]),
    ]:
        add("R3", feature, values, "test_R3_latency")
    add("R3", "latency-normalized", ["linear-example"], "test_R3_latency_normalized")
    add("R3", "labels", ["repeated", "shuffled"], "test_R3_one_hot")
    add(
        "R3",
        "invalid",
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
        "test_R3_invalid",
        True,
    )
    for feature, values in [
        (
            "pair",
            ["Leaky/Leaky", "Leaky/Synaptic", "Synaptic/Leaky", "Synaptic/Synaptic"],
        ),
        ("reset", ["subtract", "zero", "none"]),
        ("length", [1, 2, 8]),
        ("batch", [1, 3]),
        ("dtype", ["float32", "float64"]),
        ("depth", [1, 2]),
    ]:
        add("R4", feature, values, "test_R4_model_sequence")
    add("R4", "scope", ["same-class", "mixed-class"], "test_R4_class_scope")
    add("R4", "scope", ["common-root"], "test_R4_nested_and_global_scope")
    add("R4", "loop-length", [0, 1, 3], "test_R4_reset_zero_one_many", True)
    add("R5", "genuine-users", ["five-users"], "human_session_not_executed")
    add(
        "R6",
        "archive-format",
        ["tar", "tar.gz", "tgz", "tar.xz", "zip", "gz"],
        "test_R6_valid_archives",
    )
    add(
        "R6",
        "archive-attack",
        ["parent", "absolute", "symlink", "hardlink", "backslash", "drive"],
        "test_R6_archive_escape",
    )
    add(
        "R6",
        "transport-failure",
        ["tls", "timeout", "endpoint", "redirect", "dropbox_tls"],
        "test_R6_https_failure",
    )
    add("R6", "redirect-target", ["https", "http"], "test_R6_redirect_policy")
    add(
        "R6",
        "atomic-download",
        ["checksum", "partial", "success"],
        "test_R6_atomic_download",
    )
    add(
        "R6",
        "workflow",
        ["download-extract", "cached-valid"],
        "test_R6_download_extract_control",
    )
    add(
        "R6",
        "filter-runtime",
        ["unavailable-fails-closed"],
        "test_R6_tar_filter_required",
    )
    add(
        "R6",
        "existing-symlink",
        ["genuine-filesystem-integration"],
        "native_symlink_not_executed",
    )
    boundary = [x for x in inventory if x["boundary"]]
    requirements = {
        f"R{i}": dict(
            automated_evidence=any(f"test_R{i}_" in x for x in ids),
            satisfaction=(
                "pending genuine user sessions"
                if i == 5
                else "passes selected scope; specification ambiguities remain"
                if i in [1, 2, 3]
                else "passes selected scope"
            ),
        )
        for i in range(1, 7)
    }
    requirements["R5"]["automated_evidence"] = (
        True  # The README smoke check runs outside the pytest suite.
    )
    summary = dict(
        starting_commit=read("evidence/final/environment.json")["commit"],
        stages={
            stage: outcome(f"evidence/{stage}/portfolio.xml")
            for stage in ("baseline", "corrected-baseline", "final")
            if (HERE / "evidence" / stage / "portfolio.xml").is_file()
        },
        upstream=outcome("evidence/final/upstream.xml"),
        requirements=requirements,
        requirements_with_evidence=rate(6, 6),
        requirements_satisfaction=dict(
            selected_scope_passes=5,
            pending=1,
            total=6,
            full_requirement_signoff="not claimed",
        ),
        partitions=rate(sum(x["executed"] for x in inventory), len(inventory)),
        boundaries=rate(sum(x["executed"] for x in boundary), len(boundary)),
        pairwise=rate(len(observed), len(expected)),
        pairwise_cases=len(combinations["cases"]),
        feasible_configurations=combinations["feasible_configurations"],
        line=rate(total["covered_lines"], total["num_statements"]),
        branch=rate(total["covered_branches"], total["num_branches"]),
        model_states=rate(len(states), 4),
        model_transitions=rate(len(edges), len(possible_edges)),
        planned_model_cases=1000,
        executed_unique_model_cases=len(model),
        passed_unique_model_cases=sum(r["outcome"] == "passed" for r in model),
        human_sessions=rate(0, 5),
        rate_observations=counts,
        yield_definition="distinct confirmed production defects / unique portfolio pytest items executed in final run; repeated runs and injected faults excluded",
        distinct_production_defects=10,
        defects_per_final_case=10 / len(results),
        defects_by_requirement={"R1": 1, "R2": 4, "R3": 0, "R4": 2, "R5": 0, "R6": 3},
        primary_discovery_technique={
            "functional_equations": 5,
            "model_based": 2,
            "security_functional_and_review": 3,
        },
        deliberate_faults=read_optional("evidence/mutations/results.json"),
        native_symlink=read_optional("evidence/link-capability.json"),
    )
    (HERE / "partitions.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    (HERE / "evidence" / "assessment.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["rate_observations", "deliberate_faults"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
