"""
Checks that the log files in data/ are exactly the files that were captured.

    python scripts/verify_evidence.py --create   # record the fingerprints (once, right after capture)
    python scripts/verify_evidence.py            # check the files still match them

The fingerprints are SHA-256 hashes stored in evidence/SHA256SUMS, which is kept in git,
so the commit date shows when they were recorded. The same file works on Linux, so the
original copies in the VM can be checked with:  sha256sum -c SHA256SUMS
"""

from pathlib import Path
import argparse
import sys

# Make sure the project root is on sys.path so the 'src' imports work when this is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.integrity import hash_files, read_manifest, write_manifest

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = PROJECT_ROOT / "evidence" / "SHA256SUMS"


def evidence_files() -> list[Path]:
    """The captured evidence: every log (.txt) and timeline (.log) directly in data/."""
    return sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix in {".txt", ".log"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or verify SHA-256 fingerprints of the log files.")
    parser.add_argument("--create", action="store_true", help="Write evidence/SHA256SUMS from the current files")
    args = parser.parse_args()

    current = hash_files(evidence_files())

    if args.create:
        if MANIFEST_PATH.exists():
            print(f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)} already exists; delete it first to re-create it.")
            raise SystemExit(1)
        write_manifest(MANIFEST_PATH, current)
        print(f"Recorded {len(current)} fingerprints in {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
        return

    if not MANIFEST_PATH.exists():
        print("No evidence/SHA256SUMS yet. Run with --create first.")
        raise SystemExit(1)

    recorded = read_manifest(MANIFEST_PATH)
    all_ok = True
    for name, digest in sorted(recorded.items()):
        if name not in current:
            status = "MISSING"
        elif current[name] == digest:
            status = "OK"
        else:
            status = "CHANGED"
        all_ok &= status == "OK"
        print(f"  {status:8s} {name}  {digest[:16]}...")
    for name in sorted(set(current) - set(recorded)):
        print(f"  {'NEW':8s} {name}  (not in the manifest)")

    print("\nAll evidence files match their recorded fingerprints." if all_ok
          else "\nWARNING: at least one evidence file does not match its recorded fingerprint.")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
