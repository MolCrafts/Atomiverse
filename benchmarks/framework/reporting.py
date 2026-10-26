"""One result bundle -> wide method table, detailed CSV, roadmap and CI snapshot."""

from __future__ import annotations

import csv
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .comparison import evaluate
from .execution import write_json
from .provenance import file_hash
from .specification import Bundle


def roadmap_status(bundle: Bundle) -> list[dict]:
    status = []
    catalog = bundle.protocol.get("catalog", {}).get("capabilities", [])
    catalog_present = bool(catalog)
    if not catalog:
        catalog = [{"id": r.capability, "rows": [r.id], "cpu": "implemented",
                    "cuda": "implemented"} for r in bundle.rows]
    for capability in catalog:
        for backend in ("cpu", "cuda"):
            row_ids = set(capability.get(f"{backend}_rows", capability["rows"]))
            policy = bundle.protocol.get("coverage_policy", {})
            qualification_tiers = {"acceptance", "performance"} if bundle.tier == "performance" else {"acceptance"}
            expected = {rid: [s["id"] for s in policy.get(rid, {}).get("systems", [])
                              if qualification_tiers.intersection(s["tiers"])] for rid in row_ids}
            if policy:
                active_ids = {rid for rid in row_ids if expected[rid]}
            else:
                active_ids = row_ids
            rows = [r for r in bundle.rows if r.id in active_ids]
            participants = {r.id: [p for p in r.participants if p.role == "atv" and p.backend == backend and p.available]
                            for r in rows}
            cells = [c for c in bundle.cells if c.row_id in participants
                     and c.participant_id in {p.id for p in participants[c.row_id]}]
            implemented = capability.get(backend) != "not_implemented" and any(
                p.available for ps in participants.values() for p in ps)
            full_rows = bool(active_ids) and active_ids == {r.id for r in rows}
            if policy:
                full_rows = full_rows and row_ids <= policy.keys() and all(
                    set(expected[r.id]) == {s.id for s in r.systems} for r in rows)
            full_scope = catalog_present and full_rows and bundle.tier in ("acceptance", "performance")
            accepted = bool(cells) and all(c.numerical == "pass" for c in cells)
            qualified = backend != "cuda" or all(c.cuda_only == "pass" for c in cells)
            receipt = bundle.protocol.get("build_receipt") or {}
            verified_binary = bool(receipt) and receipt.get("source_tree_sha256") == bundle.source.get("tree_sha256") and all(
                o.metadata.get("binary_sha256") == receipt.get("binary_sha256")
                for c in cells for o in c.observations if o.status == "completed")
            official = verified_binary and not bundle.source["dirty"] and bundle.source.get("build_verified", False) and not bundle.source.get("changed_during_run") and not bundle.protocol["selection_partial"]
            mark = "○" if capability.get(backend) == "not_implemented" else "◐"
            if any(c.numerical == "fail" for c in cells):
                mark = "✗"
            elif implemented and full_scope and accepted and qualified and official:
                mark = "⚡" if bundle.tier == "performance" and all(c.speedup is not None for c in cells) else "✓"
            reasons = []
            if capability.get(backend) == "not_implemented":
                reasons.append("not implemented")
            else:
                if not full_scope:
                    reasons.append("required coverage incomplete")
                if not accepted:
                    reasons.append("numerical evidence missing or failed")
                if backend == "cuda" and not qualified:
                    reasons.append("CUDA-only trace qualification incomplete")
                if not official:
                    reasons.append("official source/build qualification incomplete")
                if not capability["rows"] and capability.get("reason"):
                    reasons.append(capability["reason"])
            status.append({"capability_id": capability["id"], "row_ids": sorted(row_ids), "backend": backend,
                           "label": capability.get("label", capability["id"]),
                           "chapter": capability.get("chapter"), "declared_scope": capability.get("scope"),
                           "mark": mark, "numerical_pass": accepted, "cuda_only_qualified": qualified,
                           "official_source": official, "campaign_id": bundle.campaign_id,
                           "scope": list(dict.fromkeys(s.id for r in rows for s in r.systems)),
                           "coverage_complete": full_scope,
                           "reason": "; ".join(reasons) or "complete evidence"})
    return status


def roadmap_markdown(bundle: Bundle, chapter: str | None = None) -> str:
    states = roadmap_status(bundle)
    grouped = {}
    for state in states:
        if chapter is None or state["chapter"] == chapter:
            grouped.setdefault(state["capability_id"], {})[state["backend"]] = state
    if not grouped:
        raise ValueError("no benchmark catalog entries map to this roadmap chapter")
    lines = [f"Campaign `{bundle.campaign_id}`; profile `{bundle.profile}`; tier `{bundle.tier}`.", "",
             "Numerical acceptance and roadmap qualification are separate. A local dirty or "
             "unverified build cannot promote ✓/⚡. Unmigrated capabilities retain explicit gaps.", "",
             "| Capability ID | Capability | CPU | CUDA | Evidence |",
             "| --- | --- | :---: | :---: | --- |"]
    for capability_id, entries in grouped.items():
        cpu, gpu = entries["cpu"], entries["cuda"]
        evidence = "CPU numerical pass" if cpu["numerical_pass"] else "CPU numerical coverage pending"
        evidence += "; CUDA numerical pass" if gpu["numerical_pass"] else "; CUDA numerical coverage pending"
        if cpu["reason"] != "complete evidence":
            evidence += "; " + cpu["reason"]
        if gpu["reason"] not in ("complete evidence", "not implemented", cpu["reason"]):
            evidence += "; " + gpu["reason"]
        fields = [f"`{capability_id}`", cpu["label"], cpu["mark"], gpu["mark"], evidence]
        lines.append("| " + " | ".join(f.replace("|", "\\|").replace("\n", " ") for f in fields) + " |")
    return "\n".join(lines) + "\n"


def matrix(bundle: Bundle) -> dict:
    columns = list(dict.fromkeys(p.id for row in bundle.rows for p in row.participants))
    lookup = {(c.row_id, c.system_id, c.participant_id): c for c in bundle.cells}
    rows = []
    for row in bundle.rows:
        values = {}
        for participant in columns:
            entries = []
            for system in row.systems:
                cell = lookup.get((row.id, system.id, participant))
                if cell is None:
                    continue
                time = cell.timing.get("median_seconds")
                error_groups = {}
                for check in cell.checks:
                    if "max_abs_error" in check:
                        key = check["observable"], check["unit"]
                        error_groups.setdefault(key, []).append(check)
                errors = [f"{key[0]} Δmax={max(c['max_abs_error'] for c in checks):.3g} "
                          f"(limit {min(c['max_limit'] for c in checks):.3g}) {key[1]}"
                          for key, checks in error_groups.items()]
                text = f"{system.id}: {cell.numerical if cell.status == 'completed' else cell.status}"
                if time is not None:
                    text += f"; {time:.6g} s [{cell.timing['min_seconds']:.6g}, {cell.timing['max_seconds']:.6g}]"
                if errors:
                    text += "; " + ", ".join(dict.fromkeys(errors))
                if cell.speedup is not None:
                    text += f"; {cell.speedup:.3g}× vs {row.reference}"
                elif time is not None:
                    text += "; ratio ineligible: " + str(cell.speedup_reason)
                if cell.reason:
                    text += "; " + cell.reason
                entries.append(text)
            values[participant] = entries or ["not applicable"]
        rows.append({"id": row.id, "label": row.label, "systems": [s.label for s in row.systems],
                     "values": values, "notes": f"H1; FP64; reference={row.reference}"})
    return {"columns": columns, "rows": rows}


def report(bundle: Bundle, out: Path, *, snapshot: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    evaluate(bundle)
    bundle.roadmap = roadmap_status(bundle)
    table = matrix(bundle)
    analysis = {"reporter_sha256": file_hash(__file__),
                "comparison_sha256": file_hash(Path(__file__).with_name("comparison.py"))}
    payload = {"schema": bundle.schema_version, "campaign_id": bundle.campaign_id,
               "analysis": analysis,
               "summary": bundle.summary, "matrix": table, "environment": bundle.environment,
               "source": {k: v for k, v in bundle.source.items() if k != "source_files"},
               "protocol": bundle.protocol, "threads": bundle.threads, "repeats": bundle.repeats,
               "warmup": bundle.warmup, "started": bundle.started, "roadmap": bundle.roadmap,
               "results": [c.model_dump() for c in bundle.cells]}
    hardware = bundle.environment
    gpu_note = hardware["gpus"] or ("none reported" if hardware.get("gpu_query_available") else "unknown (GPU query unavailable)")
    notes = (f"H1: CPU={hardware['cpu_model']}; physical cores={hardware['physical_cores']}; "
             f"logical CPUs={hardware['logical_cpus']}; requested OMP/BLAS threads={bundle.threads}; "
             f"GPU={gpu_note}; host RAM={hardware['host_memory_bytes']} bytes; "
             f"OS={hardware['os']}; Python={hardware['python']}; {bundle.repeats} warm samples, "
             f"{bundle.warmup} warmup(s)/sample; {bundle.started}. "
             "Actual engine/compiler/threadpool versions and memory budgets are in the observations.")
    headers = ["Method / model / task", "Systems", *table["columns"], "Notes: hardware / environment"]
    lines = ["# Scientific benchmark", "", f"Commit: {bundle.source['commit']}; "
             f"dirty={bundle.source['dirty']}; campaign={bundle.campaign_id}.", "",
             "Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. "
             "Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.", "",
             "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in table["rows"]:
        fields = [row["label"], "; ".join(row["systems"]), *["<br>".join(row["values"][c]) for c in table["columns"]], row["notes"]]
        lines.append("| " + " | ".join(f.replace("|", "\\|").replace("\n", " ") for f in fields) + " |")
    lines.extend(["", notes, "", "Summary: `" + json.dumps(bundle.summary, sort_keys=True) + "`.", ""])
    (out / "matrix.md").write_text("\n".join(lines))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in headers)
    body = []
    for row in table["rows"]:
        fields = [[row["label"]], row["systems"], *[row["values"][c] for c in table["columns"]], [row["notes"]]]
        body.append("<tr>" + "".join("<td>" + "<br>".join(html.escape(v) for v in f) + "</td>" for f in fields) + "</tr>")
    (out / "matrix.html").write_text("<!doctype html><meta charset='utf-8'><title>Scientific benchmark</title>"
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:10px;vertical-align:top}th{background:#eee}</style>"
        f"<h1>Scientific benchmark</h1><p>{html.escape(str(bundle.source['commit']))}; dirty={bundle.source['dirty']}</p>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table><p>{html.escape(notes)}</p>")
    with (out / "results.long.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "system_id", "participant_id", "required", "execution", "numerical", "phase", "repeat", "seconds", "process_seconds", "median_seconds", "speedup", "reason"])
        for c in bundle.cells:
            for o in c.observations or [None]:
                writer.writerow([c.row_id, c.system_id, c.participant_id, c.required, c.status,
                                 c.numerical, o.phase if o else None, o.repeat if o else None,
                                 o.seconds if o else None, o.process_seconds if o else None,
                                 c.timing.get("median_seconds"), c.speedup, c.reason or c.speedup_reason])
    write_json(out / "roadmap-status.json", bundle.roadmap)
    (out / "roadmap-status.md").write_text(roadmap_markdown(bundle))
    suite = ET.Element("testsuite", name="Scientific benchmark", tests=str(len(bundle.cells)),
                       failures=str(sum(c.required and c.numerical != "pass" for c in bundle.cells)),
                       skipped=str(sum(not c.required and c.numerical != "pass" for c in bundle.cells)))
    for c in bundle.cells:
        case = ET.SubElement(suite, "testcase", classname=c.row_id, name=f"{c.system_id}/{c.participant_id}")
        if c.required and c.numerical != "pass":
            ET.SubElement(case, "failure", message=c.reason or c.numerical)
        elif not c.required and c.numerical != "pass":
            ET.SubElement(case, "skipped", message=c.reason or c.numerical)
    ET.ElementTree(suite).write(out / "junit.xml", encoding="unicode", xml_declaration=True)
    write_json(out / "results.json", bundle.model_dump())
    write_json(out / "payload.json", payload)
    if snapshot:
        import molci as mci
        snap = mci.Snapshot(manifest=mci.Manifest(record="benchmark", observation_id=bundle.campaign_id,
            source=mci.Source(repository="MolCrafts/Atomiverse", commit=bundle.source["commit"], timestamp=bundle.started),
            producer="atomiverse-benchmarks", profile=bundle.profile,
            tracking=mci.Tracking(enabled=False, generation=1),
            provenance={"bundle_sha256": file_hash(out / "results.json"), "analysis": analysis}), payload=payload)
        write_json(out / "snapshots/benchmark.json", snap.model_dump(mode="json"))
