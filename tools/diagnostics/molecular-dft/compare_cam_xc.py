"""Retain full CAM local-field differences, including unqualified spin boundaries."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--pins", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
pins = json.loads(args.pins.read_text())
records = []
normal_pass = True
short_range_pass = True
for i, point in enumerate(pins["records"]):
    result = subprocess.run(
        [str(args.binary.resolve()), *[repr(x) for x in point["input"]]],
        text=True,
        capture_output=True,
        check=True,
    )
    native = json.loads(result.stdout)
    error = [abs(a - b) for a, b in zip(native["output"], point["output"])]
    passing = [e <= 1e-11 * max(1.0, abs(r)) for e, r in zip(error, point["output"])]
    sr_error = []
    sr_passing = []
    for spin in (0, 1):
        reference = pins["short_range_records"][2 * i + spin]["output"]
        errors = [abs(a - b) for a, b in zip(native["short_range"][spin], reference)]
        sr_error.append(errors)
        sr_passing.extend(
            e <= 1e-11 * max(1.0, abs(r)) for e, r in zip(errors, reference)
        )
    # Diagnose the full polarized oracle against its independently executed
    # spin-scaled unpolarized ITYH calls. No model or acceptance reference is
    # changed: these are the same exchange functional evaluated two ways.
    sa, sb = native["short_range"]
    native_components = {
        "GGA_X_B88": native["b88"],
        "GGA_C_LYP": native["lyp"],
        "LDA_C_VWN": native["vwn"],
        "GGA_X_ITYH": [sa[0] + sb[0], sa[1], sb[1], sa[2], 0.0, sb[2]],
    }
    oa, ob = (pins["short_range_records"][2 * i + s]["output"] for s in (0, 1))
    scaled_ityh = [oa[0] + ob[0], oa[1], ob[1], oa[2], 0.0, ob[2]]
    ityh_scaling_error = [
        abs(a - b) for a, b in zip(scaled_ityh, point["components"]["GGA_X_ITYH"])
    ]
    component_error = {
        key: [abs(a - b) for a, b in zip(values, point["components"][key])]
        for key, values in native_components.items()
    }
    if point["qualified_normal_domain"]:
        normal_pass &= all(passing)
    short_range_pass &= all(sr_passing)
    records.append(
        {
            "input": point["input"],
            "qualified_normal_domain": point["qualified_normal_domain"],
            "native": native,
            "oracle": point["output"],
            "absolute_errors": error,
            "all_channels_passed": all(passing),
            "channel_passed": passing,
            "short_range_errors": sr_error,
            "short_range_passed": all(sr_passing),
            "oracle_components": point["components"],
            "native_components": native_components,
            "component_absolute_errors": component_error,
            "oracle_ityh_spin_scaling_errors": ityh_scaling_error,
        }
    )
evidence = {
    "normal_domain_passed": normal_pass,
    "short_range_component_passed": short_range_pass,
    "all_points_passed": all(p["all_channels_passed"] for p in records),
    "normal_point_count": sum(p["qualified_normal_domain"] for p in records),
    "boundary_point_count": sum(not p["qualified_normal_domain"] for p in records),
    "gates": {"absolute": 1e-11, "relative": 1e-11, "all_six_local_channels": True},
    "probe_binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
    "pin_file_sha256": hashlib.sha256(args.pins.read_bytes()).hexdigest(),
    "oracle_provenance": {
        k: v for k, v in pins.items() if k not in ("records", "short_range_records")
    },
    "records": records,
}
args.out.write_text(json.dumps(evidence, indent=2) + "\n")
print(
    "normal points",
    evidence["normal_point_count"],
    "pass",
    normal_pass,
    "30 ITYH samples pass",
    short_range_pass,
    "all full-CAM boundary channels pass",
    evidence["all_points_passed"],
)
raise SystemExit(0 if normal_pass and short_range_pass else 1)
