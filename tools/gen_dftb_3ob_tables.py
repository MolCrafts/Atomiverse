#!/usr/bin/env python3
"""Transcribe DFTB.org 3ob-3-1 H/C/N/O .skf → static C++ header.

Manual only — NEVER wired into CMake/ctest. In-tree data only at runtime
(no .skf I/O). SKF grammar, 10-channel order, nGrid = fileN − 1, and the
Spline repulsive layout follow tools/gen_dftb_mio_tables.py parse_skf /
expand_tokens. This generator emits a whole header; it does not patch
element rows in place.

Usage (run from the repository root):

    python3 tools/gen_dftb_3ob_tables.py [--sk-dir DIR] [--out PATH]
    python3 tools/gen_dftb_3ob_tables.py --check [--sk-dir DIR] [--out PATH]

Default --sk-dir walks up from this file to .../software/dftb-params/3ob-3-1.
Missing pair → hard error, no fallback. Output is byte-deterministic:
identical inputs produce identical bytes, and --check exits 0 iff the
committed header matches the regenerated text.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

ELEMENTS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}
Z_BY_SYM = {v: k for k, v in ELEMENTS.items()}

# Existing committed pair order (homos, then X–H/H–X, then remaining heteros).
PAIR_ORDER = (
    (1, 1),
    (6, 6),
    (7, 7),
    (8, 8),
    (6, 1),
    (1, 6),
    (7, 1),
    (1, 7),
    (8, 1),
    (1, 8),
    (6, 7),
    (7, 6),
    (6, 8),
    (8, 6),
    (7, 8),
    (8, 7),
)

# SKF 10-channel order (DFTB+): ddσ ddπ ddδ pdσ pdπ ppσ ppπ sdσ spσ ssσ
IDX_PPS, IDX_PPP, IDX_SPS, IDX_SSS = 5, 6, 8, 9

DEFAULT_OUT = Path("include/atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h")

# H–H pin from 3ob-3-1 H-H.skf line 3 (first SK integral row, ssσ).
_HH_FIRST_HSS = -5.856190768489e-01


def expand_tokens(line: str) -> list[float]:
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    raw = [t for t in re.split(r"[\s,]+", line) if t]
    out: list[float] = []
    for t in raw:
        m = re.fullmatch(r"(\d+)\*(.+)", t)
        if m:
            out.extend([float(m.group(2))] * int(m.group(1)))
        else:
            out.append(float(t))
    return out


def parse_skf(path: Path) -> dict:
    lines = path.read_text().splitlines()
    try:
        spline_i = next(
            i for i, l in enumerate(lines) if l.strip().startswith("Spline")
        )
    except StopIteration as exc:
        raise RuntimeError(f"{path.name}: no Spline section") from exc
    head = lines[0].replace(",", " ").split()
    grid_dist = float(head[0])
    n_file = int(float(head[1]))
    n_grid = n_file - 1  # DFTB+ convention (matches dftb_mio_hh_sk.h)

    sym_a, sym_b = path.stem.split("-")
    za, zb = Z_BY_SYM[sym_a], Z_BY_SYM[sym_b]
    homo = za == zb

    # DFTB+ OldSKData_readFromFile (oldskdata.F90):
    #   homo:   grid; atomic; mass+polyRep header; then nGrid SK integral lines
    #   hetero: grid; polyRep header; then nGrid SK integral lines
    # nGrid = fileN − 1 for new-style interpolation.
    atomic = None
    if homo:
        atomic = expand_tokens(lines[1])
        # lines[2] = mass + polynomial repulsive header — not an SK integral row.
        sk_rows = []
        for l in lines[3:spline_i]:
            v = expand_tokens(l)
            sk_rows.append((v + [0.0] * 20)[:20])
    else:
        # lines[1] = polynomial repulsive header — not an SK integral row.
        sk_rows = []
        for l in lines[2:spline_i]:
            v = expand_tokens(l)
            sk_rows.append((v + [0.0] * 20)[:20])

    if len(sk_rows) < n_grid:
        raise RuntimeError(f"{path.name}: only {len(sk_rows)} SK rows, need {n_grid}")
    sk_rows = sk_rows[:n_grid]

    def col(i: int) -> list[float]:
        return [row[i] for row in sk_rows]

    sk_points = [
        (
            row[IDX_SSS],
            row[IDX_SPS],
            row[IDX_PPS],
            row[IDX_PPP],
            row[10 + IDX_SSS],
            row[10 + IDX_SPS],
            row[10 + IDX_PPS],
            row[10 + IDX_PPP],
        )
        for row in sk_rows
    ]

    # Repulsive spline
    n_spline = int(expand_tokens(lines[spline_i + 1])[0])
    rep_cutoff = expand_tokens(lines[spline_i + 1])[1]
    exp_coeffs = expand_tokens(lines[spline_i + 2])
    if len(exp_coeffs) != 3:
        raise RuntimeError(f"{path.name}: expected 3 exp coeffs, got {exp_coeffs}")
    x_start: list[float] = []
    sp_coeffs: list[float] = []  # 4 * (n_spline - 1)
    sp_last: list[float] = []
    rep_int: list[list[float]] = []
    for k in range(n_spline):
        vals = expand_tokens(lines[spline_i + 3 + k])
        x_start.append(vals[0])
        if k + 1 < n_spline:
            # x0 x1 c0 c1 c2 c3
            if len(vals) < 6:
                raise RuntimeError(f"{path.name}: short cubic interval {k}: {vals}")
            sp_coeffs.extend(vals[2:6])
            coeffs = list(vals[2:6]) + [0.0, 0.0]
        else:
            # x0 x1 + 6 last coeffs
            if len(vals) < 8:
                raise RuntimeError(f"{path.name}: short last interval: {vals}")
            sp_last = vals[2:8]
            coeffs = list(vals[2:8])
        if len(vals) < 2:
            raise RuntimeError(f"{path.name}: spline interval {k} missing x1")
        rep_int.append([vals[0], vals[1]] + coeffs)

    return {
        "za": za,
        "zb": zb,
        "sym_a": sym_a,
        "sym_b": sym_b,
        "homo": homo,
        "atomic": atomic,
        "grid_dist": grid_dist,
        "n_grid": n_grid,
        "H_sss": col(IDX_SSS),
        "H_sps": col(IDX_SPS),
        "H_pps": col(IDX_PPS),
        "H_ppp": col(IDX_PPP),
        "S_sss": col(10 + IDX_SSS),
        "S_sps": col(10 + IDX_SPS),
        "S_pps": col(10 + IDX_PPS),
        "S_ppp": col(10 + IDX_PPP),
        "sk_points": sk_points,
        "n_spline": n_spline,
        "n_rep_int": n_spline,
        "rep_cutoff": rep_cutoff,
        "exp_coeffs": exp_coeffs,
        "x_start": x_start,
        "sp_coeffs": sp_coeffs,
        "sp_last": sp_last,
        "rep_int": rep_int,
    }


def fmt_d(v: float) -> str:
    """17 significant digits, no float suffix."""
    return f"{v:.16e}"


def pair_tag(za: int, zb: int) -> str:
    return f"{ELEMENTS[za]}{ELEMENTS[zb]}"


def default_sk_dir() -> Path:
    # tools/ → repo → molcrafts → software/dftb-params/3ob-3-1
    here = Path(__file__).resolve()
    cand = here.parents[2] / "software" / "dftb-params" / "3ob-3-1"
    if cand.is_dir():
        return cand
    cand2 = here.parents[1].parent / "software" / "dftb-params" / "3ob-3-1"
    if cand2.is_dir():
        return cand2
    return cand


def regenerate_stamp(sk_dir_cli: Path | None, out: Path) -> str:
    cmd = "python3 tools/gen_dftb_3ob_tables.py"
    if sk_dir_cli is not None:
        cmd += f" --sk-dir {sk_dir_cli.as_posix()}"
    cmd += f" --out {out.as_posix()}"
    return cmd


def element_rows(pairs: list[dict]) -> list[str]:
    by_z: dict[int, dict] = {}
    for p in pairs:
        if not p["homo"]:
            continue
        at = p["atomic"]
        if at is None or len(at) < 10:
            raise RuntimeError(f"{p['sym_a']}-{p['sym_b']}: missing homo atomic line")
        # Ed Ep Es SPE Ud Up Us occd occp occs
        ep, es = at[1], at[2]
        us = at[6]
        n_val = at[7] + at[8] + at[9]
        n_ao = 1 if p["za"] == 1 else 4
        e_p = 0.0 if n_ao == 1 else ep
        r_cut = p["n_grid"] * p["grid_dist"]
        by_z[p["za"]] = {
            "n_ao": n_ao,
            "n_val": n_val,
            "es": es,
            "ep": e_p,
            "us": us,
            "r_cut": r_cut,
        }
    rows = []
    for z, sym in ELEMENTS.items():
        if z not in by_z:
            raise RuntimeError(f"missing homo SKF for Z={z} ({sym})")
        d = by_z[z]
        rows.append(
            "    {"
            f'{z}, "{sym}", {d["n_ao"]}, {d["n_val"]:.1f}, '
            f"{fmt_d(d['es'])}, {fmt_d(d['ep'])}, {fmt_d(d['us'])}, "
            f"{fmt_d(d['r_cut'])}"
            "},"
        )
    return rows


def emit_header(pairs: list[dict], stamp: str) -> str:
    by_ab = {(p["za"], p["zb"]): p for p in pairs}
    missing = [ab for ab in PAIR_ORDER if ab not in by_ab]
    extra = [ab for ab in by_ab if ab not in PAIR_ORDER]
    if missing or extra:
        raise RuntimeError(f"pair set mismatch: missing={missing} extra={extra}")

    lines: list[str] = [
        "/// @file cpu/semiempirical/basis_data/dftb_3ob_tables.h",
        "/// @brief Static DFTB 3ob-3-1 SK tables for organic H/C/N/O (SCC-DFTB2 path).",
        "///",
        "/// **GENERATED** — do not edit by hand.",
        f"/// Regenerate: {stamp}",
        "///",
        "/// Transcribed offline from DFTB.org 3ob-3-1 `.skf` files (CC-BY-SA 4.0;",
        "/// Gaus, Goez, Elstner, J. Chem. Theory Comput. 2013, 9, 338–354;",
        "/// DOI 10.1021/ct300849w). In-tree data only — nothing is parsed at",
        "/// build, test, or run time.",
        "///",
        "/// Units inside this header: length [Bohr], energy [Ha] as in the original",
        "/// SKF; DFTB2 converts to eV/Å on the public surface. Grid convention",
        "/// matches DFTB+ (nGrid = fileN − 1 points at r = i * gridDist).",
        "///",
        "/// Deferred layout: 3ob stays point-major (`Dftb3obSkPoint`); mio stays",
        "/// channel-major. Unifying the two is postponed to spec 1 (runtime SKF",
        '/// reader). Do not "fix" the layouts in passing.',
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace ATV {",
        "namespace cpu {",
        "namespace semiempirical {",
        "namespace basis_data {",
        "",
        "/// @brief One 3ob element (onsite + Hubbard) in Hartree.",
        "struct Dftb3obElement {",
        "    int z;",
        "    const char* sym;",
        "    int n_ao;       ///< 1 (s) or 4 (s+p)",
        "    double n_val;   ///< valence e",
        "    double e_s;     ///< [Ha]",
        "    double e_p;     ///< [Ha]",
        "    double hubbard_u; ///< [Ha] shell U (Us)",
        "    double r_cut;   ///< electronic cutoff [Bohr]",
        "};",
        "",
        "/// @brief Radial SK channels at one grid point (Ha / dimensionless).",
        "struct Dftb3obSkPoint {",
        "    double h_ss, h_sp, h_pps, h_ppp;",
        "    double s_ss, s_sp, s_pps, s_ppp;",
        "};",
        "",
        "/// @brief One ordered pair radial table + repulsive spline.",
        "struct Dftb3obPair {",
        "    int z_a, z_b;",
        "    double grid;          ///< [Bohr]",
        "    int n_grid;",
        "    const Dftb3obSkPoint* sk; ///< n_grid points",
        "    double rep_cutoff;    ///< [Bohr]",
        "    double rep_a1, rep_a2, rep_a3; ///< short-range exp(-a1*r+a2)+a3",
        "    int n_rep_int;",
        "    const double* rep_int; ///< flat: n_rep_int * (start,end,c0..c3[,c4,c5]) with stride 8",
        "};",
        "",
        "inline constexpr Dftb3obElement kDftb3obElements[] = {",
    ]
    lines.extend(element_rows(pairs))
    lines.append("};")
    lines.append("inline constexpr std::size_t kDftb3obNElements =")
    lines.append("    sizeof(kDftb3obElements) / sizeof(kDftb3obElements[0]);")
    lines.append("")

    ordered = [by_ab[ab] for ab in PAIR_ORDER]
    for p in ordered:
        tag = pair_tag(p["za"], p["zb"])
        lines.append(f"inline constexpr Dftb3obSkPoint kDftb3obSk_{tag}[] = {{")
        for pt in p["sk_points"]:
            inner = ", ".join(fmt_d(v) for v in pt)
            lines.append(f"    {{{inner}}},")
        lines.append("};")
        lines.append(f"inline constexpr double kDftb3obRep_{tag}[] = {{")
        for iv in p["rep_int"]:
            inner = ", ".join(fmt_d(v) for v in iv)
            lines.append(f"    {inner},")
        lines.append("};")

    lines.append("inline constexpr Dftb3obPair kDftb3obPairs[] = {")
    for p in ordered:
        tag = pair_tag(p["za"], p["zb"])
        a1, a2, a3 = p["exp_coeffs"]
        lines.append(
            "    {"
            f"{p['za']}, {p['zb']}, {fmt_d(p['grid_dist'])}, {p['n_grid']}, "
            f"kDftb3obSk_{tag}, {fmt_d(p['rep_cutoff'])}, "
            f"{fmt_d(a1)}, {fmt_d(a2)}, {fmt_d(a3)}, "
            f"{p['n_rep_int']}, kDftb3obRep_{tag}"
            "},"
        )
    lines.append("};")
    lines.append("inline constexpr std::size_t kDftb3obNPairs =")
    lines.append("    sizeof(kDftb3obPairs) / sizeof(kDftb3obPairs[0]);")
    lines.append("")
    lines.append("} // namespace basis_data")
    lines.append("} // namespace semiempirical")
    lines.append("} // namespace cpu")
    lines.append("} // namespace ATV")
    lines.append("")
    return "\n".join(lines)


def load_pairs(sk_dir: Path) -> list[dict]:
    if not sk_dir.is_dir():
        raise SystemExit(f"error: SK dir not found: {sk_dir} — no fallback")
    pairs = []
    for za in ELEMENTS:
        for zb in ELEMENTS:
            name = f"{ELEMENTS[za]}-{ELEMENTS[zb]}.skf"
            path = sk_dir / name
            if not path.is_file():
                raise SystemExit(f"error: missing {path} — no fallback")
            pairs.append(parse_skf(path))
    hh = next(p for p in pairs if p["za"] == 1 and p["zb"] == 1)
    if hh["n_grid"] != 549:
        raise SystemExit(f"error: H–H n_grid={hh['n_grid']}, expected 549")
    if abs(hh["grid_dist"] - 0.02) > 1e-15:
        raise SystemExit(f"error: H–H grid={hh['grid_dist']}, expected 0.02")
    first = hh["sk_points"][0][0]
    if abs(first - _HH_FIRST_HSS) > 1e-15:
        raise SystemExit(f"error: H–H first Hss={first!r}, expected {_HH_FIRST_HSS!r}")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and diff against --out; exit 0 iff identical",
    )
    ap.add_argument("--sk-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    sk_dir = args.sk_dir or default_sk_dir()
    stamp = regenerate_stamp(args.sk_dir, args.out)
    pairs = load_pairs(sk_dir)
    text = emit_header(pairs, stamp)

    if args.check:
        if not args.out.is_file():
            print(f"error: {args.out} does not exist", file=sys.stderr)
            return 1
        on_disk = args.out.read_text(encoding="utf-8")
        if text == on_disk:
            print(f"OK: {args.out} is byte-identical to regenerated output")
            return 0
        diff = difflib.unified_diff(
            on_disk.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=str(args.out),
            tofile="regenerated",
        )
        sys.stdout.writelines(diff)
        print(f"MISMATCH: {args.out} differs from regenerated output")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.out} ({args.out.stat().st_size // 1024} KiB, {len(pairs)} pairs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
