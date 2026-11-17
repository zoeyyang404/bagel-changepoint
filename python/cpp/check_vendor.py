"""Verify that an isolated source distribution has its complete C++ core."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "vendor"
manifest = json.loads((root / "core-sha256.json").read_text())
for name, digest in manifest.items():
    data = (root / "bagel" / name).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"Core checksum mismatch: {name}")
print(f"Verified {len(manifest)} bundled core files.")
