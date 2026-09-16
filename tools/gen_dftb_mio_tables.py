#!/usr/bin/env python3
"""Transcribe DFTB.org mio-1-1 H/C/N/O .skf → static C++ headers.

In-tree data only at runtime (no .skf I/O). Matches the existing H–H pin
convention: DFTB+ nGrid = fileN − 1, polyInterU1 / distFudge handled in C++,
repulsive = exp head + cubic intervals + 5th-order last segment.

Usage:
  python3 tools/gen_dftb_mio_tables.py \\
      [--sk-dir /path/to/mio-1-1] \\
      [--out-sk include/.../dftb_mio_sk_tables.h] \\
      [--out-elements include/.../dftb_mio_tables.h]

Default --sk-dir walks up from this file to .../software/dftb-params/mio-1-1.
"""

from __future__ import annotations

import argparse
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

# SKF 10-channel order (DFTB+): ddσ ddπ ddδ pdσ pdπ ppσ ppπ sdσ spσ ssσ
IDX_PPS, IDX_PPP, IDX_SPS, IDX_SSS = 5, 6, 8, 9

HA_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903


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
    spline_i = next(i for i, l in enumerate(lines) if l.strip().startswith("Spline"))
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

    # Repulsive spline
    n_spline = int(expand_tokens(lines[spline_i + 1])[0])
    rep_cutoff = expand_tokens(lines[spline_i + 1])[1]
    exp_coeffs = expand_tokens(lines[spline_i + 2])
    if len(exp_coeffs) != 3:
        raise RuntimeError(f"{path.name}: expected 3 exp coeffs, got {exp_coeffs}")
    x_start: list[float] = []
    sp_coeffs: list[float] = []  # 4 * (n_spline - 1)
    sp_last: list[float] = []
    for k in range(n_spline):
        vals = expand_tokens(lines[spline_i + 3 + k])
        x_start.append(vals[0])
        if k + 1 < n_spline:
            # x0 x1 c0 c1 c2 c3
            if len(vals) < 6:
                raise RuntimeError(f"{path.name}: short cubic interval {k}: {vals}")
            sp_coeffs.extend(vals[2:6])
        else:
            # x0 x1 + 6 last coeffs
            if len(vals) < 8:
                raise RuntimeError(f"{path.name}: short last interval: {vals}")
            sp_last = vals[2:8]

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
        "n_spline": n_spline,
        "rep_cutoff": rep_cutoff,
        "exp_coeffs": exp_coeffs,
        "x_start": x_start,
        "sp_coeffs": sp_coeffs,
        "sp_last": sp_last,
    }


def fmt_array(name: str, values: list[float], per_line: int = 4) -> str:
    parts = [f"inline constexpr double {name}[] = {{"]
    line: list[str] = []
    for i, v in enumerate(values):
        line.append(f"{v:.16e}")
        if len(line) == per_line or i + 1 == len(values):
            parts.append(
                "    " + ", ".join(line) + ("," if i + 1 < len(values) else "")
            )
            line = []
    parts.append("};")
    return "\n".join(parts)


def pair_tag(za: int, zb: int) -> str:
    return f"{ELEMENTS[za]}{ELEMENTS[zb]}"


def emit_sk_header(pairs: list[dict], out: Path) -> None:
    lines = [
        "/// @file cpu/semiempirical/basis_data/dftb_mio_sk_tables.h",
        "/// @brief Static mio-1-1 Slater–Koster + repulsive tables (H/C/N/O).",
        "///",
        "/// **GENERATED** by `tools/gen_dftb_mio_tables.py` from DFTB.org /",
        "/// dftbparams mio-1-1 `*.skf` (Elstner et al., Phys. Rev. B 58, 7260",
        "/// (1998)). Do not edit by hand — re-run the generator.",
        "///",
        "/// In-tree data only — no runtime `.skf` parser. SK grid distances in",
        "/// **Bohr**; integral energies in **Hartree** (DFTB+ convention).",
        "/// Public ATV conversion to eV/Å is done in `internal/mio_sk_table.h`.",
        "///",
        "/// Column channels (sp subset of the SKF 10-pack): ssσ, spσ, ppσ, ppπ.",
        "/// nGrid = fileN − 1 (DFTB+); distFudge handled at evaluation time.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace ATV {",
        "namespace cpu {",
        "namespace semiempirical {",
        "namespace basis_data {",
        "",
        "/// @brief Hartree → eV (CODATA-consistent with DFTB+ printout).",
        f"inline constexpr double kMioSk_HaToEV = {HA_TO_EV:.12f};",
        "/// @brief Bohr → Å.",
        f"inline constexpr double kMioSk_BohrToA = {BOHR_TO_A:.12f};",
        "/// @brief Å → Bohr.",
        "inline constexpr double kMioSk_AToBohr = 1.0 / kMioSk_BohrToA;",
        "/// @brief New-style SK interpolation fudge [Bohr] (DFTB+ distFudge).",
        "inline constexpr double kMioSk_distFudge_Bohr = 1.0;",
        "",
        "/// @brief One ordered element pair of transcribed mio-1-1 tables.",
        "struct MioSkPairData {",
        "    int za;  ///< atomic number of first centre (filename left)",
        "    int zb;  ///< atomic number of second centre (filename right)",
        "    std::size_t n_grid;",
        "    double grid_dist_bohr;",
        "    const double* H_sss;  ///< H ssσ [Ha]",
        "    const double* H_sps;  ///< H spσ [Ha] (s on A, p on B)",
        "    const double* H_pps;  ///< H ppσ [Ha]",
        "    const double* H_ppp;  ///< H ppπ [Ha]",
        "    const double* S_sss;",
        "    const double* S_sps;",
        "    const double* S_pps;",
        "    const double* S_ppp;",
        "    std::size_t n_spline;",
        "    double rep_cutoff_bohr;",
        "    double exp_coeffs[3];",
        "    const double* x_start_bohr;     ///< n_spline starts [Bohr]",
        "    const double* sp_coeffs;       ///< 4 × (n_spline − 1) cubic",
        "    const double* sp_last_coeffs;  ///< 6 coeffs of last 5th-order piece",
        "};",
        "",
    ]

    for p in pairs:
        tag = pair_tag(p["za"], p["zb"])
        lines.append(f"// ── {p['sym_a']}–{p['sym_b']} ──────────────────────────────")
        for ch in (
            "H_sss",
            "H_sps",
            "H_pps",
            "H_ppp",
            "S_sss",
            "S_sps",
            "S_pps",
            "S_ppp",
        ):
            lines.append(fmt_array(f"kMioSk_{tag}_{ch}", p[ch]))
            lines.append("")
        lines.append(f"inline constexpr double kMioSk_{tag}_xStart[] = {{")
        lines.append("    " + ", ".join(f"{v:.16e}" for v in p["x_start"]) + ",")
        lines.append("};")
        lines.append("")
        lines.append(fmt_array(f"kMioSk_{tag}_spCoeffs", p["sp_coeffs"]))
        lines.append("")
        lines.append(
            f"inline constexpr double kMioSk_{tag}_spLast[6] = {{"
            + ", ".join(f"{v:.16e}" for v in p["sp_last"])
            + "};"
        )
        lines.append("")

    lines.append("/// @brief All ordered H/C/N/O mio-1-1 pairs (16 = 4×4).")
    lines.append("inline constexpr MioSkPairData kMioSkPairs[] = {")
    for p in pairs:
        tag = pair_tag(p["za"], p["zb"])
        ec = p["exp_coeffs"]
        lines.append(
            f"    {{{p['za']}, {p['zb']}, {p['n_grid']}u, {p['grid_dist']:.16e},"
        )
        lines.append(
            f"     kMioSk_{tag}_H_sss, kMioSk_{tag}_H_sps, kMioSk_{tag}_H_pps, kMioSk_{tag}_H_ppp,"
        )
        lines.append(
            f"     kMioSk_{tag}_S_sss, kMioSk_{tag}_S_sps, kMioSk_{tag}_S_pps, kMioSk_{tag}_S_ppp,"
        )
        lines.append(f"     {p['n_spline']}u, {p['rep_cutoff']:.16e},")
        lines.append(f"     {{{ec[0]:.16e}, {ec[1]:.16e}, {ec[2]:.16e}}},")
        lines.append(
            f"     kMioSk_{tag}_xStart, kMioSk_{tag}_spCoeffs, kMioSk_{tag}_spLast}},"
        )
    lines.append("};")
    lines.append(
        "inline constexpr std::size_t kMioSkNPairs ="
        " sizeof(kMioSkPairs) / sizeof(kMioSkPairs[0]);"
    )
    lines.append("")
    lines.append("} // namespace basis_data")
    lines.append("} // namespace semiempirical")
    lines.append("} // namespace cpu")
    lines.append("} // namespace ATV")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {len(pairs)} pairs)")


def emit_elements_patch(pairs: list[dict], tables_h: Path) -> None:
    """Rewrite element rows in dftb_mio_tables.h from homo SKF atomic lines."""
    by_z = {}
    for p in pairs:
        if p["homo"] and p["atomic"] is not None:
            at = p["atomic"]
            # Ed Ep Es SPE Ud Up Us occd occp occs
            ep, es = at[1], at[2]
            us = at[6]
            n_val = at[7] + at[8] + at[9]
            by_z[p["za"]] = {
                "es_ha": es,
                "ep_ha": ep,
                "u_ha": us,
                "n_val": n_val,
            }

    # r_cut [Å] = max pair SK cutoff involving Z
    r_cut = {z: 0.0 for z in ELEMENTS}
    for p in pairs:
        # grid end + fudge, same as MioSkTable::cutoff
        c_A = (p["n_grid"] * p["grid_dist"] + 1.0) * BOHR_TO_A
        # also rep cutoff
        c_A = max(c_A, p["rep_cutoff"] * BOHR_TO_A)
        r_cut[p["za"]] = max(r_cut[p["za"]], c_A)
        r_cut[p["zb"]] = max(r_cut[p["zb"]], c_A)

    text = tables_h.read_text()
    # Replace kDftbElements block
    rows = []
    for z, sym in ELEMENTS.items():
        d = by_z[z]
        n_ao = 1 if z == 1 else 4
        es = d["es_ha"] * HA_TO_EV
        ep = 0.0 if z == 1 else d["ep_ha"] * HA_TO_EV
        u = d["u_ha"] * HA_TO_EV
        rows.append(
            f'    {{{z}, "{sym}", {n_ao}, {d["n_val"]:.1f}, {es:.12f}, {ep:.12f}, '
            f"{u:.12f}, {r_cut[z]:.6f}}},"
        )
    new_block = (
        "inline constexpr DftbElementRow kDftbElements[] = {\n"
        + "\n".join(rows)
        + "\n};"
    )
    text2, n = re.subn(
        r"inline constexpr DftbElementRow kDftbElements\[\] = \{.*?\n\};",
        new_block,
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError("failed to rewrite kDftbElements in dftb_mio_tables.h")

    # File prologue is owned by the committed header (U^d / W are not in
    # this table). Do not rewrite it here.
    tables_h.write_text(text2)
    print(f"updated element rows in {tables_h}")
    for z, sym in ELEMENTS.items():
        d = by_z[z]
        print(
            f"  {sym}: Es={d['es_ha'] * HA_TO_EV:.6f} eV  "
            f"U={d['u_ha'] * HA_TO_EV:.6f} eV  r_cut={r_cut[z]:.3f} Å"
        )


def default_sk_dir() -> Path:
    # tools/ → repo → molcrafts → software/dftb-params/mio-1-1
    here = Path(__file__).resolve()
    cand = here.parents[2] / "software" / "dftb-params" / "mio-1-1"
    if cand.is_dir():
        return cand
    # fallback: sibling of Atomiverse
    cand2 = here.parents[1].parent / "software" / "dftb-params" / "mio-1-1"
    if cand2.is_dir():
        return cand2
    return cand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sk-dir", type=Path, default=None)
    ap.add_argument(
        "--out-sk",
        type=Path,
        default=Path(
            "include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_sk_tables.h"
        ),
    )
    ap.add_argument(
        "--out-elements",
        type=Path,
        default=Path(
            "include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
        ),
    )
    ap.add_argument(
        "--skip-elements",
        action="store_true",
        help="Only emit SK tables header; do not patch dftb_mio_tables.h",
    )
    args = ap.parse_args()
    sk_dir = args.sk_dir or default_sk_dir()
    if not sk_dir.is_dir():
        print(f"SK dir not found: {sk_dir}", file=sys.stderr)
        return 1

    pairs = []
    for za in ELEMENTS:
        for zb in ELEMENTS:
            name = f"{ELEMENTS[za]}-{ELEMENTS[zb]}.skf"
            path = sk_dir / name
            if not path.is_file():
                print(f"missing {path}", file=sys.stderr)
                return 1
            pairs.append(parse_skf(path))
            print(
                f"parsed {name}: n_grid={pairs[-1]['n_grid']} n_spline={pairs[-1]['n_spline']}"
            )

    # Verify H–H ssσ against legacy pin header if present
    legacy = Path("include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_hh_sk.h")
    if legacy.is_file():
        hh = next(p for p in pairs if p["za"] == 1 and p["zb"] == 1)
        text = legacy.read_text()
        m = re.search(r"kMioHH_Hss_Ha\[\] = \{([^}]+)\}", text, re.S)
        if m:
            nums = [
                float(x)
                for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?", m.group(1))
            ]
            if len(nums) != len(hh["H_sss"]):
                print(
                    f"WARN: H–H n_grid {len(hh['H_sss'])} vs legacy {len(nums)}",
                    file=sys.stderr,
                )
            else:
                max_d = max(abs(a - b) for a, b in zip(nums, hh["H_sss"]))
                print(f"H–H vs legacy dftb_mio_hh_sk.h max|ΔHss| = {max_d:.3e}")
                if max_d > 1e-12:
                    print(
                        "ERROR: H–H transcription drifted from pin table",
                        file=sys.stderr,
                    )
                    return 1

    emit_sk_header(pairs, args.out_sk)
    if not args.skip_elements:
        emit_elements_patch(pairs, args.out_elements)
    return 0


if __name__ == "__main__":
    sys.exit(main())
