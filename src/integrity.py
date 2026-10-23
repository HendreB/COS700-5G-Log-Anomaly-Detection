"""
Evidence integrity for the log files.

In a forensic setting the analysis must never change the evidence. My pipeline only
opens the logs for reading, but I also want to be able to *show* that. So every input
file gets a SHA-256 fingerprint before a run and again after it: if a single byte had
changed, the fingerprint would be different.

The manifest (evidence/SHA256SUMS) uses the same format as the Linux 'sha256sum' tool,
so the original files in the VM can be checked against it with 'sha256sum -c'.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so big logs don't have to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    """Fingerprint each file, keyed by file name."""
    return {Path(p).name: sha256_file(p) for p in paths}


def check_unchanged(paths: list[Path], before: dict[str, str]) -> bool:
    """Re-hash the inputs after a run and report whether any of them changed."""
    after = hash_files(paths)
    changed = [name for name, digest in before.items() if after.get(name) != digest]
    if changed:
        print(f"WARNING: input files changed during the run: {', '.join(changed)}")
        return False
    print(f"Evidence integrity: {len(before)} input file(s) unchanged (SHA-256 before = after).")
    return True


def write_manifest(manifest_path: Path, hashes: dict[str, str]) -> None:
    """Write the fingerprints in sha256sum format: '<hash>  <file name>'."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(manifest_path: Path) -> dict[str, str]:
    """Read a sha256sum-format manifest back into {file name: hash}."""
    hashes = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            hashes[name.lstrip("*")] = digest
    return hashes
