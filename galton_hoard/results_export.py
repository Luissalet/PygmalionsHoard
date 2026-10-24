"""Download stored result metadata without running models or including their answers."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

# One ordered schema for both formats. Names carry units in CSV; JSON also describes them.
COLUMNS = (
    ("id", ""), ("run", ""), ("contestant", ""), ("contestant_name", ""), ("suite", ""),
    ("case", ""), ("title", ""), ("category", ""), ("repeat", ""), ("score", ""),
    ("passed", ""), ("skipped", ""), ("unavailable", ""), ("truncated", ""), ("cpu", ""),
    ("judge_pending", ""), ("self_judged", ""), ("error", ""), ("latency_ms", "ms"),
    ("ttft_ms", "ms"), ("decode_tps", "tokens/s"), ("prompt_tokens", "tokens"),
    ("completion_tokens", "tokens"), ("detail", ""),
)


def prepare(svc, run, model, only_failed):
    started = datetime.fromtimestamp(svc.clock(), timezone.utc).isoformat()
    names = {c["id"]: c["name"] for c in svc.store.contestants()}
    rows, offset = [], 0
    while True:
        page = svc.store.results(run_id=run["id"], contestant_id=model, limit=500, offset=offset, with_output=False)
        titles = svc.case_titles([r["case_id"] for r in page])
        for row in page:
            # Match RunDetail's visible filter, including pending/unavailable rows.
            if not only_failed or (not row["passed"] and not row["skipped"]):
                rows.append(svc.result_card(row, titles, names, output=False))
        offset += len(page)
        if len(page) < 500:
            break
    metadata = {
        "run": run["id"], "model": model, "only_failed": only_failed,
        "state_at_start": run["state"], "count": len(rows),
        "read_started_at": started,
        "read_finished_at": datetime.fromtimestamp(svc.clock(), timezone.utc).isoformat(),
    }
    columns = [{"name": name, "unit": unit} for name, unit in COLUMNS]
    json_text = json.dumps({**metadata, "columns": columns, "results": rows}, ensure_ascii=False, allow_nan=False)
    # Repeat capture criteria in each CSV record, keeping an ordinary rectangular table.
    capture = {"export_model": model, "export_only_failed": only_failed, "export_count": len(rows),
               "export_state_at_start": run["state"], "export_read_started_at": started,
               "export_read_finished_at": metadata["read_finished_at"]}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[*capture, *(name for name, _ in COLUMNS)], lineterminator="\r\n")
    writer.writeheader()
    writer.writerows({**capture, **row, "detail": json.dumps(row["detail"], ensure_ascii=False, allow_nan=False)} for row in rows)
    return {**metadata, "columns": columns, "csv_text": stream.getvalue(), "json_text": json_text}
