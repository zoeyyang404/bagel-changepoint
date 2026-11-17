"""Copy the shared C++ files into the R and Python packages.

Run from any directory: python tools/sync_core.py [--check]
Keep the copies in version control so each package can be built from its
own source archive, without downloading the rest of the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "core-files.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report drift without writing files")
    args = parser.parse_args()
    names = [line.strip() for line in MANIFEST.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    targets = [ROOT / "bagelR" / "src", ROOT / "python" / "vendor" / "bagel"]
    changed = []
    hashes = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError(f"Expected a plain filename in {MANIFEST}: {name}")
        data = (ROOT / "c++" / name).read_bytes().replace(b"\r\n", b"\n")
        hashes[name] = hashlib.sha256(data).hexdigest()
        for target in targets:
            path = target / name
            if not path.exists() or path.read_bytes().replace(b"\r\n", b"\n") != data:
                changed.append(str(path.relative_to(ROOT)))
                if not args.check:
                    target.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
    manifest_path = ROOT / "python" / "vendor" / "core-sha256.json"
    data = (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode()
    if not manifest_path.exists() or manifest_path.read_bytes() != data:
        changed.append(str(manifest_path.relative_to(ROOT)))
        if not args.check:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(data)
    if changed:
        print(("Out of sync:\n" if args.check else "Updated:\n") + "\n".join(changed))
    else:
        print(f"All {len(names)} core files match in the R and Python packages.")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
