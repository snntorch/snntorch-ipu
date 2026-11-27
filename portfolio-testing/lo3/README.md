# LO3 tests and results

This folder contains the tests, runner and reports for the snnTorch testing portfolio. The [LO3 report](LO3_Testing_Report.pdf) records the Windows CPU run. To generate fresh evidence, run the commands below from the repository root on the `lo3-testing` branch in a Python 3.12 virtual environment.

```sh
python -m pip install -e . pytest coverage scipy tqdm h5py requests torchvision six
python portfolio-testing/lo3/run.py final
python portfolio-testing/lo3/mutations.py
python portfolio-testing/lo3/assess.py
```

- `run.py` runs the repository tests, the LO3 tests with branch coverage, and the quickstart and training smoke checks. Look in `evidence/final` for output, environment details and `commands.json`, which lists the subprocess exit codes.
- `mutations.py` runs three deliberate changes in separate processes and checks whether the relevant tests fail. Its logs are in `evidence/mutations`.
- `assess.py` summarises the recorded results and recomputes pairwise coverage from case IDs. It still matches test names for some partition and boundary counts, so check individual case IDs before treating those totals as verified.
- `generate.py` regenerates the case manifests from the declared design and seeds.
- `original_security.py` replays the controlled dataset checks against the starting downloader revision. Use a checkout with that commit in its history. Test failures are expected in this replay.

The runners stop if their output directory already exists. Use a fresh checkout for a repeat run so the earlier evidence stays available.

## How to read the case files

| File | What it records |
|---|---|
| `design.json` | The test scope, tolerances, rate-test settings and assumptions about firing and reset. |
| `pairwise.json` | The settings varied in the neuron tests and the configurations chosen to cover every pair. |
| `model-cases.json` | The 1,000 reset cases, including the model pair, settings and seed for each case. |
| `rate-intervals.json` | The input probability, seed, draw count and acceptable spike count for each rate check. |
| `partitions.json` | The reported input categories and boundaries, with their linked test names and results. |
