"""Build a portable, offline HTML canvas artifact from completed reproduction results."""
import base64
import csv
import hashlib
import html
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/eikonal_five_seed_device3"


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    if not summary["complete"] or summary["evaluations"] != summary["expected_evaluations"]:
        raise ValueError("Artifact requires a completed reproduction")
    suites = {}
    for row in csv.DictReader((RESULTS / "suite_per_seed.csv").open()):
        entry = suites.setdefault(row["cell"], {}).setdefault(row["model"], {})
        for key, col in [("success", "suite_success_pct"), ("collision", "suite_collision_pct"),
                         ("nonconv", "suite_nonconvergence_pct")]:
            entry.setdefault(key, []).append((int(row["training_seed"]), float(row[col])))
    for models in suites.values():
        for metrics in models.values():
            for key, values in list(metrics.items()):
                values.sort()
                numbers = [v for _, v in values]
                metrics[key] = dict(n=len(numbers), mean=statistics.mean(numbers),
                                    sd=statistics.stdev(numbers), values=numbers,
                                    seeds=[seed for seed, _ in values])
    for cell, models in suites.items():
        for model, metrics in models.items():
            assert abs(metrics["success"]["mean"] - summary["suites"][cell][model]["mean"]) < 1e-10
    report = (RESULTS / "REPORT.html").read_text()
    body = re.search(r"<main>(.*)</main>", report, re.S).group(1)
    style = re.search(r"<style>(.*?)</style>", report, re.S).group(1)
    resources = ["per_seed.csv", "suite_per_seed.csv", "summary.json", "manifest.json", "REPORT.md"]
    types = {".csv": "text/csv", ".json": "application/json", ".md": "text/markdown"}
    downloads = []
    for name in resources:
        uri = "data:" + types[Path(name).suffix] + ";base64," + base64.b64encode((RESULTS / name).read_bytes()).decode()
        link = f'href="{uri}" download="{name}"'
        body = body.replace(f'href="{name}"', link)
        downloads.append(f'<a {link}>{html.escape(name)}</a>')
    for ref in re.findall(r'src="([^"]+)"', body):
        path = RESULTS / ref
        if path.suffix != ".svg":
            raise ValueError(f"Unexpected figure: {ref}")
        body = body.replace(f'src="{ref}"', 'src="data:image/svg+xml;base64,' + base64.b64encode(path.read_bytes()).decode() + '"')
    payload = dict(suites=suites, sets=summary["sets"], contrasts=summary["suite_comparisons"])
    template = (Path(__file__).with_name("artifact_template.html")).read_text()
    for key, value in {"STYLE": style, "FULL_REPORT": body, "DOWNLOADS": " · ".join(downloads),
                       "DATA": json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")}.items():
        template = template.replace("@@" + key + "@@", value)
    target = RESULTS / "CANVAS_REPORT.html"
    target.write_text(template)
    record = dict(artifact=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  complete=True, training_runs=90, evaluations=summary["evaluations"],
                  native_canvas=False, rendering="HTML Canvas API", self_contained=True,
                  inputs={name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
                          for name in resources + ["REPORT.html"]})
    (RESULTS / "artifact_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Created {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
