"""One CLI for scientific benchmarks. No aliases for the removed regression runner."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import tomllib

from .framework.execution import run_campaign, worker, write_json
from .framework.provenance import source_info
from .framework.reporting import report, roadmap_markdown
from .framework.specification import Bundle, Method

ROOT = Path(__file__).resolve().parents[1]


def methods() -> list[Method]:
    specs = []
    for path in sorted((ROOT / "benchmarks").glob("*/bench_*.py")):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        spec = getattr(importlib.import_module(module), "SPEC", None)
        if not isinstance(spec, Method):
            raise TypeError(f"{path}: expected a Method SPEC")
        specs.append(spec)
    if len({s.id for s in specs}) != len(specs):
        raise ValueError("duplicate method IDs")
    return specs


def catalog_check(specs: list[Method]) -> dict:
    path = ROOT / "benchmarks/roadmap.toml"
    data = tomllib.loads(path.read_text())
    if data.get("schema") != "atomiverse.benchmark-catalog/1":
        raise ValueError("unsupported roadmap catalog schema")
    capabilities = data["capabilities"]
    lookup = {c["id"]: c for c in capabilities}
    if len(lookup) != len(capabilities):
        raise ValueError("duplicate capability IDs")
    rows = {r.id: r for m in specs for r in m.rows}
    for row in rows.values():
        if row.capability not in lookup or row.id not in lookup[row.capability]["rows"]:
            raise ValueError(f"{row.id}: missing roadmap mapping")
    for capability in capabilities:
        if not (ROOT / "docs/roadmap" / capability["chapter"]).is_file():
            raise ValueError(f"missing roadmap chapter: {capability['chapter']}")
        for row_id in capability["rows"]:
            if row_id not in rows or rows[row_id].capability != capability["id"]:
                raise ValueError(f"{row_id}: orphan or conflicting catalog row")
    return data


def run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Write local scientific results without the molci CI snapshot envelope")
    parser.add_argument("--method", action="append", help="method ID; repeat to select several")
    parser.add_argument("--tier", choices=("smoke", "acceptance", "performance"), default="smoke")
    parser.add_argument("--system", action="append", help="explicit system IDs (partial coverage)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, help="warm samples; performance defaults to five")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--require-cuda", action="store_true", help="require every declared implemented ATV CUDA workload")
    parser.add_argument("--source-manifest", type=Path, help="verified source identity for a transported immutable snapshot")
    parser.add_argument("--build-receipt", type=Path, help="attested CMake/source/native-binary identity")
    parser.add_argument("--out", type=Path, required=True)


def execute(args: argparse.Namespace, specs: list[Method]) -> int:
    if args.threads < 1 or args.warmup < 0 or args.timeout <= 0 or (args.repeats is not None and args.repeats < 1):
        raise ValueError("invalid thread, sample, warmup or timeout budget")
    selected = [s for s in specs if not args.method or s.id in args.method]
    unknown = set(args.method or []) - {s.id for s in specs}
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if args.system:
        known = {s.id for m in selected for r in m.rows for s in r.systems}
        if set(args.system) - known:
            raise ValueError("unknown system selection")
    catalog = catalog_check(methods())
    bundle = run_campaign(selected, tier=args.tier, repeats=args.repeats or (5 if args.tier == "performance" else 1),
                          warmup=args.warmup, threads=args.threads, timeout=args.timeout,
                          out=args.out, root=ROOT, require_cuda=args.require_cuda,
                          system_ids=args.system, catalog=catalog, source_manifest=args.source_manifest,
                          build_receipt=args.build_receipt)
    if args.no_snapshot:
        bundle.protocol["ci_snapshot"] = "disabled explicitly by --no-snapshot"
    report(bundle, args.out.resolve(), snapshot=not args.no_snapshot)
    print(json.dumps(bundle.summary, sort_keys=True))
    print(f"Table: {args.out.resolve() / 'matrix.html'}")
    return 0 if bundle.summary["complete"] else 1


def method_main(spec: Method) -> int:
    parser = argparse.ArgumentParser(description=f"Scientific benchmark: {spec.id}")
    run_arguments(parser)
    args = parser.parse_args()
    args.method = [spec.id]
    return execute(args, [spec])


def main() -> int:
    parser = argparse.ArgumentParser(description="ATV scientific benchmarks: accuracy, speed and resources")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("catalog")
    source = sub.add_parser("source", help="capture source hashes before transporting a snapshot")
    source.add_argument("--out", type=Path, required=True)
    attest = sub.add_parser("attest-build", help="build with CMake and record matching source/native fingerprints")
    attest.add_argument("--build-dir", type=Path, required=True)
    attest.add_argument("--out", type=Path, required=True)
    attest.add_argument("--jobs", type=int, default=2)
    attest.add_argument("--source-manifest", type=Path)
    run_arguments(sub.add_parser("run"))
    for action in ("report", "validate"):
        item = sub.add_parser(action)
        item.add_argument("--input", type=Path, required=True)
        if action == "report":
            item.add_argument("--out", type=Path, required=True)
            item.add_argument("--no-snapshot", action="store_true",
                              help="Write local results without the molci CI snapshot envelope")
    roadmap = sub.add_parser("roadmap", help="check or update the generated evidence block")
    roadmap.add_argument("--input", type=Path, required=True)
    roadmap.add_argument("--chapter", default="hartree-fock.md")
    mode = roadmap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    internal = sub.add_parser("worker", help=argparse.SUPPRESS)
    internal.add_argument("--request", type=Path, required=True)
    internal.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args.request, args.result)
    if args.command == "source":
        write_json(args.out, source_info(ROOT))
        return 0
    if args.command == "attest-build":
        from .framework.build import attest_build

        if args.jobs < 1:
            raise ValueError("build jobs must be positive")
        attest_build(ROOT, args.build_dir, args.out, jobs=args.jobs, source_manifest=args.source_manifest)
        return 0
    if args.command == "roadmap":
        from .framework.comparison import evaluate

        bundle = evaluate(Bundle.model_validate_json(args.input.read_text()))
        path = (ROOT / "docs/roadmap" / args.chapter).resolve()
        if path.parent != (ROOT / "docs/roadmap").resolve():
            raise ValueError("roadmap chapter must be directly under docs/roadmap")
        text = path.read_text()
        start, end = "<!-- benchmark-evidence:start -->", "<!-- benchmark-evidence:end -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("roadmap chapter needs exactly one benchmark evidence block")
        before, _, tail = text.partition(start)
        _, _, after = tail.partition(end)
        updated = before + start + "\n\n" + roadmap_markdown(bundle, args.chapter) + "\n" + end + after
        if args.check:
            if updated != text:
                raise ValueError("roadmap evidence differs from the supplied benchmark bundle")
        else:
            path.write_text(updated)
        return 0
    if args.command in ("report", "validate"):
        bundle = Bundle.model_validate_json(args.input.read_text())
        if args.command == "report":
            if args.no_snapshot:
                bundle.protocol["ci_snapshot"] = "disabled explicitly by --no-snapshot"
            report(bundle, args.out, snapshot=not args.no_snapshot)
        else:
            print(f"Valid {bundle.schema_version}: {len(bundle.cells)} declared cells")
        return 0
    specs = methods()
    if args.command == "catalog":
        data = catalog_check(specs)
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "list":
        for spec in specs:
            print(f"{spec.id}: {', '.join(r.id for r in spec.rows)}")
        return 0
    catalog_check(specs)
    return execute(args, specs)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"benchmark: {error}", file=sys.stderr)
        raise SystemExit(2) from error
