"""Compose the proven residual corrections after the CatBoost cold-start path.

No table or model is refit here.  The builder copies the frozen train-only
correction tables from the 1031-BSS package onto the 1016.637 CatBoost package.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "artifacts" / "submit_r10_cathetero.zip"
DEFAULT_CORRECTIONS = ROOT / "artifacts" / "submit_corrections.zip"
DEFAULT_OUT = ROOT / "artifacts" / "submit_cathetero_corrections.zip"


def copy_member(source: zipfile.ZipFile, member: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read(member))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    for path in (args.base, args.corrections):
        if not path.exists():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="cathetero_corrections_") as name:
        work = Path(name)
        with zipfile.ZipFile(args.base) as base_zip:
            base_zip.extractall(work)
        with zipfile.ZipFile(args.corrections) as correction_zip:
            copy_member(correction_zip, "model/corrections.json",
                        work / "model" / "corrections.json")
            copy_member(correction_zip, "corrections_runtime.py",
                        work / "corrections_runtime.py")

        correction_spec = json.loads(
            (work / "model" / "corrections.json").read_text(encoding="utf-8"))
        cold_spec = json.loads(
            (work / "model" / "coldstart_meta.json").read_text(encoding="utf-8"))
        if not cold_spec.get("catboost_file"):
            raise SystemExit("base package has no CatBoost cold-start expert")
        if len(correction_spec.get("items", [])) != 3:
            raise SystemExit("expected exactly three frozen residual corrections")

        script_path = work / "script.py"
        script = script_path.read_text(encoding="utf-8")
        import_anchor = "from coldstart_runtime import apply_coldstart_expert\n"
        if script.count(import_anchor) != 1:
            raise SystemExit("unexpected cold-start import anchor")
        script = script.replace(
            import_anchor,
            import_anchor + "from corrections_runtime import apply_corrections\n",
            1,
        )
        call = 'apply_coldstart_expert(preds, test, "./model")'
        member = "\n        return " + call
        plain = "\n    return " + call
        if script.count(member) != 1 or script.count(plain) != 1:
            raise SystemExit("unexpected cold-start return anchors")
        script = script.replace(
            member,
            "\n        preds = " + call
            + '\n        return apply_corrections(preds, test, "./model")',
            1,
        )
        script = script.replace(
            plain,
            "\n    preds = " + call
            + '\n    return apply_corrections(preds, test, "./model")',
            1,
        )
        script_path.write_text(script, encoding="utf-8")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as output:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    output.write(path, path.relative_to(work))
    print(f"built {args.out}")
    print("pipeline: CatBoost cold-start -> beta/gamma/delta corrections")


if __name__ == "__main__":
    main()
