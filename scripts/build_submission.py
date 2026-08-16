"""Build dist/submit.zip from the versioned submission directory."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission"
OUTPUT = ROOT / "dist" / "submit.zip"

with (SOURCE / "model" / "meta.json").open(encoding="utf-8") as f:
    meta = json.load(f)

model_files = meta.get("all_model_files", meta["model_files"])
files = [SOURCE / "script.py", SOURCE / "requirements.txt", SOURCE / "model" / "meta.json"]
files.extend(SOURCE / "model" / name for name in model_files)

OUTPUT.parent.mkdir(exist_ok=True)
with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, path.relative_to(SOURCE))

print(f"Built {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.2f} MiB)")
