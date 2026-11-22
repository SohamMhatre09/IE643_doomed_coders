"""register_result.py -- appends / replaces one finished run in <NA_OUT>/all_runs.csv (mean / sd of AS, SE, SP, HS)."""
import csv
import json
import os

import na_paths as P

METRICS = ["AS", "SE", "SP", "HS"]
HEADER = ["id", "method", "command", "result_folder"] + [m + s for m in METRICS for s in ("_mean", "_std")]


def register(run_id, family, method, script, folder, summary_json):
    path = summary_json if os.path.isabs(summary_json) else os.path.join(P.OUT, summary_json)
    if not os.path.exists(path):
        path = os.path.join(P.OUT, "ast_runs", os.path.basename(os.path.dirname(path)), "final_summary.json")
    d = json.load(open(path))
    row = {"id": run_id, "method": method, "command": script, "result_folder": os.path.dirname(path)}
    for m in METRICS:
        row[m + "_mean"], row[m + "_std"] = "%.2f" % d[m]["mean"], "%.2f" % d[m]["std"]
    csv_path = os.path.join(P.OUT, "all_runs.csv")
    rows = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else []
    rows = [r for r in rows if r["id"] != run_id] + [row]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
