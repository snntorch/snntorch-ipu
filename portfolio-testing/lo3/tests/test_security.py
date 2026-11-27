"""Check extraction and download handling with temporary files and simulated requests."""

import hashlib
import io
import os
import ssl
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile

import pytest
from snntorch.spikevision import neuromorphic_dataset as ds


def canary_state(path):
    s = path.stat()
    return dict(bytes=path.read_bytes().hex(), size=s.st_size, mtime_ns=s.st_mtime_ns)


@pytest.mark.parametrize(
    "kind", ["parent", "absolute", "symlink", "hardlink", "backslash", "drive"]
)
@pytest.mark.parametrize("fmt", ["tar", "zip"])
def test_R6_archive_escape(tmp_path, kind, fmt, evidence):
    dest = tmp_path / "out"
    dest.mkdir()
    canary = tmp_path / "canary.txt"
    canary.write_bytes(b"UNCHANGED")
    before = canary_state(canary)
    names = {
        "parent": "../canary.txt",
        "absolute": str(canary.resolve()),
        "symlink": "escape",
        "hardlink": "escape",
        "backslash": "..\\canary.txt",
        "drive": str(canary.resolve()),
    }
    # Even paths with a Windows drive point to the temporary check file.
    archive = tmp_path / ("attack." + fmt)
    if fmt == "tar":
        with tarfile.open(archive, "w") as f:
            info = tarfile.TarInfo(names[kind])
            if kind in ["symlink", "hardlink"]:
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                info.linkname = str(canary.resolve())
                f.addfile(info)
            else:
                info.size = 7
                f.addfile(info, io.BytesIO(b"CHANGED"))
    else:
        with zipfile.ZipFile(archive, "w") as f:
            info = zipfile.ZipInfo(names[kind])
            if kind in ["symlink", "hardlink"]:
                # ZIP has no hard-link entry, so this fixture uses the POSIX link form.
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                f.writestr(info, str(canary.resolve()))
            else:
                f.writestr(info, b"CHANGED")
    rejected = False
    try:
        ds._extract_archive(str(archive), str(dest))
    except (ValueError, tarfile.TarError, OSError):
        rejected = True
    after = canary_state(canary)
    evidence(
        requirement="R6",
        fixture=f"{fmt}-{kind}",
        destination=str(dest),
        before=before,
        after=after,
        rejected=rejected,
        accepted=[str(p.resolve()) for p in dest.rglob("*")],
    )
    assert before == after
    assert rejected


@pytest.mark.parametrize("fmt", ["tar", "tar.gz", "tgz", "tar.xz", "zip", "gz"])
def test_R6_valid_archives(tmp_path, fmt, evidence):
    import gzip

    archive = tmp_path / ("valid." + fmt)
    dest = tmp_path / "out"
    dest.mkdir()
    if fmt == "zip":
        with zipfile.ZipFile(archive, "w") as f:
            f.writestr("ok.txt", b"OK")
    elif fmt == "gz":
        with gzip.open(archive, "wb") as f:
            f.write(b"OK")
    else:
        mode = (
            "w:gz" if fmt in ["tar.gz", "tgz"] else "w:xz" if fmt == "tar.xz" else "w"
        )
        with tarfile.open(archive, mode) as f:
            info = tarfile.TarInfo("ok.txt")
            info.size = 2
            f.addfile(info, io.BytesIO(b"OK"))
    ds._extract_archive(str(archive), str(dest), remove_finished=True)
    files = list(dest.rglob("*"))
    evidence(
        requirement="R6",
        fixture="valid-" + fmt,
        accepted=[str(p.resolve()) for p in files],
    )
    assert len(files) == 1 and files[0].read_bytes() == b"OK" and not archive.exists()


@pytest.mark.parametrize(
    "scenario", ["tls", "timeout", "endpoint", "redirect", "dropbox_tls"]
)
def test_R6_https_failure(tmp_path, monkeypatch, scenario, evidence):
    attempts = []

    def fail(url, *args, **kwargs):
        attempts.append(str(url))
        if scenario == "tls" or scenario == "dropbox_tls":
            raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))
        if scenario == "timeout":
            raise TimeoutError("controlled timeout")
        if scenario == "redirect":
            raise urllib.error.URLError("controlled insecure redirect")
        raise urllib.error.HTTPError(url, 503, "controlled unavailable", {}, None)

    class Opener:
        def open(self, url, *args, **kwargs):
            return fail(url)

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)
    # Patch both urllib and its six alias. The starting version uses the cached alias.
    from six.moves import urllib as six_urllib

    monkeypatch.setattr(six_urllib.request, "urlretrieve", fail)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    import requests

    monkeypatch.setattr(requests, "get", fail)
    url = (
        "https://dropbox.invalid/test"
        if scenario == "dropbox_tls"
        else "https://example.invalid/test"
    )
    with pytest.raises((OSError, urllib.error.URLError)):
        ds.download_url(url, str(tmp_path), "download.bin")
    evidence(
        requirement="R6",
        fixture=scenario,
        attempts=attempts,
        files=[p.name for p in tmp_path.iterdir()],
    )
    assert attempts and all(u.startswith("https://") for u in attempts)
    assert not (tmp_path / "download.bin").exists()


def test_R6_redirect_policy(tmp_path, evidence):
    # Let the same redirect handler used by the downloader decide.
    handler = getattr(ds, "_SecureRedirect", None)
    assert handler is not None, "No HTTPS redirect guard exists"
    req = urllib.request.Request("https://example.invalid/start")
    observed = []
    for target in ["http://example.invalid/end", "https://example.invalid/end"]:
        try:
            redirected = handler().redirect_request(req, None, 302, "Found", {}, target)
            observed.append(redirected.full_url)
        except urllib.error.URLError:
            observed.append("rejected")
    evidence(requirement="R6", observed=observed)
    assert observed == ["rejected", "https://example.invalid/end"]


def test_R6_download_extract_control(tmp_path, monkeypatch, evidence):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr("ok.txt", b"OK")
    data = payload.getvalue()
    attempts = []

    def retrieve(url, path, **kwargs):
        attempts.append(url)
        with open(path, "wb") as f:
            f.write(data)

    class Response(io.BytesIO):
        headers = {}

    class Opener:
        def open(self, url, *args, **kwargs):
            attempts.append(url)
            return Response(data)

    monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
    from six.moves import urllib as six_urllib

    monkeypatch.setattr(six_urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    ds.download_and_extract_archive(
        "https://example.invalid/ok.zip",
        str(tmp_path / "download"),
        str(tmp_path / "out"),
        md5=hashlib.md5(data).hexdigest(),
    )
    assert (tmp_path / "out" / "ok.txt").read_bytes() == b"OK"
    evidence(requirement="R6", attempts=attempts, expected="OK", observed="OK")
    ds.download_url(
        "https://example.invalid/ok.zip",
        str(tmp_path / "download"),
        md5=hashlib.md5(data).hexdigest(),
    )
    assert len(attempts) == 1
