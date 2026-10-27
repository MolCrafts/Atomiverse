"""Print genuine libxc CAM-B3LYP semilocal energies and first derivatives."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyscf
from pyscf.dft import libxc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--header", type=Path, required=True)
args = parser.parse_args()
points = [
    (0.5, 0.5, [0, 0, 0], [0, 0, 0]),
    (0.3, 0.1, [0.2, 0.1, -0.1], [0.05, -0.02, 0.03]),
    (0.8, 0.2, [1, -0.3, 0.2], [-0.2, 0.1, 0.02]),
    (1e-4, 1e-5, [0.003, 0.001, -0.002], [0.0002, -0.0001, 0.0003]),
    (1e-8, 1e-8, [0, 0, 0], [0, 0, 0]),
    (3e-5, 3e-5, [0, 0, 0], [0, 0, 0]),
    (3.2e-5, 3.2e-5, [0, 0, 0], [0, 0, 0]),
    (0.4, 0, [0.2, -0.1, 0.05], [0, 0, 0]),
    (0, 0.4, [0, 0, 0], [0.2, -0.1, 0.05]),
    (0.5, 1e-8, [0.1, 0.05, -0.07], [1e-9, -1e-9, 2e-9]),
    (0.1, 0.2, [0.3, -0.2, 0.1], [-0.3, 0.2, -0.1]),
    (1e-6, 1e-8, [0.01, 0.02, 0.03], [-0.001, 0.002, -0.003]),
    (0.05, 0.03, [10, 20, 30], [3, -4, 2]),
    (10, 5, [1, -2, 3], [0.3, 0.1, -0.2]),
    (0.03, 0.04, [1e-8, 0, 0], [-1e-8, 0, 0]),
]
records = []
for ra, rb, ga, gb in points:
    rho = np.array([[[ra], *[[v] for v in ga]], [[rb], *[[v] for v in gb]]])
    exc, vxc, _, _ = libxc.eval_xc("CAM-B3LYP", rho, spin=1, deriv=1)
    vrho, vsigma = vxc[:2]
    records.append(
        {
            "input": [
                ra,
                rb,
                float(np.dot(ga, ga)),
                float(np.dot(ga, gb)),
                float(np.dot(gb, gb)),
            ],
            "qualified_normal_domain": min(ra, rb) > 0
            and min(ra, rb) / max(ra, rb) >= 1e-6,
            "output": [
                float(exc[0] * (ra + rb)),
                *vrho[0].tolist(),
                *vsigma[0].tolist(),
            ],
        }
    )
for record, (ra, rb, ga, gb) in zip(records, points):
    rho = np.array([[[ra], *[[v] for v in ga]], [[rb], *[[v] for v in gb]]])
    components = {}
    for name in ("GGA_X_B88", "GGA_X_ITYH", "LDA_C_VWN", "GGA_C_LYP"):
        ingredients = rho[:, 0, :] if name == "LDA_C_VWN" else rho
        kwargs = {"omega": 0.33} if name == "GGA_X_ITYH" else {}
        exc, vxc, _, _ = libxc.eval_xc(name, ingredients, spin=1, deriv=1, **kwargs)
        sigma = (
            vxc[1][0].tolist()
            if len(vxc) > 1 and vxc[1] is not None
            else [0.0, 0.0, 0.0]
        )
        components[name] = [float(exc[0] * (ra + rb)), *vxc[0][0].tolist(), *sigma]
    record["components"] = components
short_range = []
for ra, rb, ga, gb in points:
    for density, gradient in ((ra, ga), (rb, gb)):
        rho = np.array([[2 * density], *[[2 * v] for v in gradient]])
        exc, vxc, _, _ = libxc.eval_xc("GGA_X_ITYH", rho, spin=0, deriv=1, omega=0.33)
        short_range.append(
            {
                "input": [density, float(np.dot(gradient, gradient))],
                "output": [
                    float(exc[0] * density),
                    float(vxc[0][0]),
                    float(2 * vxc[1][0]),
                ],
            }
        )
evidence = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "command": sys.argv,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "pyscf": pyscf.__version__,
    "libxc": libxc.__version__,
    "functional": "CAM-B3LYP (libxc id433)",
    "rsh_coeff": list(libxc.rsh_coeff("CAM-B3LYP")),
    "convention": "energy per volume; derivatives in rho_alpha,rho_beta,sigma_aa,sigma_ab,sigma_bb order; no exact-exchange matrix in local XC",
    "oracle_binaries": {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(pyscf.__file__).resolve().parent.rglob("*.so"))
    },
    "records": records,
    "qualified_domain": "full CAM local pins: both spin densities positive, min/max >= 1e-6; all 15 original points retained as diagnostics",
    "short_range_component": "GGA_X_ITYH, omega=.33; one-spin values recovered from spin-unpolarized density 2*rho and sigma 4*sigma to avoid zeta cancellation",
    "short_range_records": short_range,
}
args.out.write_text(json.dumps(evidence, indent=2) + "\n")
header = [
    "// Genuine PySCF/libxc CAM-B3LYP pins; regenerate with tools/diagnostics/molecular-dft/pin_cam_xc.py.",
    "#pragma once",
    "#include <array>",
    "namespace cam_b3lyp_pin {",
    "struct Point { std::array<double,5> input; std::array<double,6> output; };",
]


def emit(name, rows, kind):
    header.append(f"inline constexpr std::array<{kind}, {len(rows)}> {name} = {{{{")
    for row in rows:
        inp = ", ".join(format(x, ".17g") for x in row["input"])
        val = ", ".join(format(x, ".17g") for x in row["output"])
        header.append("    {{{" + inp + "}}, {{" + val + "}}},")
    header.append("}};")


emit("all_points", records, "Point")
emit("points", [r for r in records if r["qualified_normal_domain"]], "Point")
header.append(
    "struct ShortRangePoint { std::array<double,2> input; std::array<double,3> output; };"
)
emit("short_range_points", short_range, "ShortRangePoint")
header += ["} // namespace cam_b3lyp_pin", ""]
args.header.write_text("\n".join(header))
print(
    "pinned",
    len(records),
    "points,",
    len(short_range),
    "ITYH spin samples; PySCF",
    pyscf.__version__,
    "libxc",
    libxc.__version__,
)
