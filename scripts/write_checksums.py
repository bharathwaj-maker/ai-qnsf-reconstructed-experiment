#!/usr/bin/env python3
"""Write SHA256SUMS.txt for every project file except the checksum file itself."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/"SHA256SUMS.txt"
rows=[]
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path==TARGET: continue
    if any(part in {".venv","venv","__pycache__","_work",".git"} for part in path.parts): continue
    if any(part.startswith("ai_qnsf_eval_chunks_") for part in path.parts): continue
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    rows.append(f"{h.hexdigest()}  {path.relative_to(ROOT).as_posix()}")
TARGET.write_text("\n".join(rows)+"\n",encoding="utf-8")
print(f"Wrote {len(rows)} SHA-256 entries to {TARGET}")
