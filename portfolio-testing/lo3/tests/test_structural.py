"""Check paths that the first coverage report showed were missing."""

import io
import os
from pathlib import Path
import tarfile
import urllib.request

import pytest
import torch
import snntorch as snn
from snntorch import backprop, functional, surrogate, utils
from snntorch.spikevision import neuromorphic_dataset as ds


@pytest.mark.parametrize("hidden", [False, True])
def test_R1_inhibition_and_reset_change(hidden):
    with pytest.warns(UserWarning, match="Inhibition"):
        n = snn.Leaky(0.5, init_hidden=hidden, output=True, inhibition=True)
    x = torch.tensor([[2.0, 3.0, 0.0], [0.0, 2.0, 4.0]])
    spk, _ = n(x) if hidden else n(x, torch.zeros_like(x))
    assert spk.tolist() == [[0, 1, 0], [0, 0, 1]]
    n.reset_mechanism = "none"
    assert n.reset_mechanism_val == 2 and n.reset_mechanism == "none"
    with pytest.raises(ValueError):
        n.reset_mechanism = "invalid"


def test_R2_direct_lso():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    surrogate.LeakySpikeOperator.apply(x, 0.25).sum().backward()
    assert x.grad.tolist() == [0.25, 0.25, 1]


def test_R4_reset_unprocessed_and_detachment():
    n = snn.Leaky(0.5, init_hidden=True)
    utils.reset(n)
    assert hasattr(n.mem, "init_flag")
    assert utils.is_leaky and not utils.is_synaptic
    n(torch.ones(1, 3, requires_grad=True))
    assert n.mem.grad_fn is not None
    utils.reset(n)
    assert n.mem.grad_fn is None and not n.mem.any()
    utils.reset(torch.nn.Identity())
    assert not utils.is_leaky


def test_R4_backprop_integration(evidence):
    # Check the flags that the training helper reads.
    torch.manual_seed(89)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        snn.Leaky(
            0.5, init_hidden=True, output=True, spike_grad=surrogate.fast_sigmoid()
        ),
    )
    optimiser = torch.optim.SGD(net.parameters(), lr=0.01)
    loader = [(torch.tensor([[0.5, 1.0]]), torch.tensor([1]))]
    loss = backprop.TBPTT(
        net,
        loader,
        optimiser,
        functional.mse_count_loss(),
        num_steps=4,
        time_var=False,
        K=2,
    )
    evidence(requirement="R4", seed=89, tbptt_loss=float(loss.detach()))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mode", ["checksum", "partial", "success"])
def test_R6_atomic_download(tmp_path, monkeypatch, mode, evidence):
    class Response(io.BytesIO):
        def read(self, *args):
            if mode == "partial":
                raise OSError("controlled incomplete response")
            return super().read(*args)

    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"OK")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    if mode == "checksum":
        with pytest.raises(RuntimeError, match="corrupted"):
            ds.download_url(
                "https://example.invalid/ok?dl=1", str(tmp_path), md5="0" * 32
            )
    elif mode == "partial":
        with pytest.raises(OSError, match="incomplete"):
            ds.download_url("https://example.invalid/ok?dl=1", str(tmp_path))
    else:
        ds.download_url("https://example.invalid/ok?dl=1", str(tmp_path))
        assert (tmp_path / "ok").read_bytes() == b"OK"
    evidence(
        requirement="R6", fixture=mode, remaining=[p.name for p in tmp_path.iterdir()]
    )
    assert not list(tmp_path.glob(".snntorch-*"))
    if mode != "success":
        assert not list(tmp_path.iterdir())


def test_R6_download_filename_escape(tmp_path):
    with pytest.raises(ValueError):
        ds.download_url("https://example.invalid/a", str(tmp_path / "out"), "../canary")
    assert not (tmp_path / "canary").exists()


def test_R6_tar_filter_required(tmp_path, monkeypatch):
    archive = tmp_path / "ok.tar"
    with tarfile.open(archive, "w") as f:
        info = tarfile.TarInfo("ok")
        info.size = 2
        f.addfile(info, io.BytesIO(b"OK"))
    monkeypatch.delattr(tarfile, "data_filter")
    with pytest.raises(RuntimeError, match="data_filter"):
        ds._extract_archive(str(archive), str(tmp_path / "out"))
    assert not (tmp_path / "out" / "ok").exists()


def test_R6_unsupported_archive(tmp_path):
    p = tmp_path / "unknown.bin"
    p.write_bytes(b"OK")
    with pytest.raises(ValueError, match="not supported"):
        ds._extract_archive(str(p))


def test_R6_portable_path_decision(tmp_path):
    for name in ["../x", "..\\x", "/x", "C:/x", "\\\\server\\share\\x"]:
        with pytest.raises(ValueError):
            ds._contained_path(str(tmp_path), name)
    assert Path(ds._contained_path(str(tmp_path), "a/b")).is_relative_to(tmp_path)


def test_R6_existing_escaping_link_resolution(tmp_path, monkeypatch):
    # Check containment after path resolution. Creating a real link is a separate check.
    original = ds.os.path.realpath
    outside = tmp_path.parent / "external-disposable-placeholder"

    def resolve(path):
        if os.path.normcase(os.path.normpath(path)) == os.path.normcase(
            str(tmp_path / "existing" / "file")
        ):
            return str(outside)
        return original(path)

    monkeypatch.setattr(ds.os.path, "realpath", resolve)
    with pytest.raises(ValueError, match="escapes"):
        ds._contained_path(str(tmp_path), "existing/file")
