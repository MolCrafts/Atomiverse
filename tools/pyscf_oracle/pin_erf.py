"""Print an independent libcint oracle for the attenuated ERI and its derivative."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyscf
from pyscf import gto

xyz = [(0.1, -0.2, 0.3), (0.8, 0.1, -0.4), (-0.2, 0.6, 0.5), (0.3, -0.5, 0.9)]
mol = gto.M(
    atom=[("H", p) for p in xyz],
    unit="Bohr",
    cart=True,
    basis={"H": [[0, [1.0, 1.0]]]},
    verbose=0,
)
with mol.with_range_coulomb(0.33):
    eri = float(mol.intor("int2e_cart")[0, 1, 2, 3])
    gradient = -mol.intor("int2e_ip1_cart")[:, 0, 1, 2, 3]
print(
    json.dumps(
        {
            "pyscf": pyscf.__version__,
            "command": sys.argv,
            "utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_executable": sys.executable,
            "oracle_binaries": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(pyscf.__file__).resolve().parent.rglob("*.so"))
            },
            "coordinates_bohr": xyz,
            "exponent_bohr_inverse_square": 1.0,
            "operator": "erf(0.33*r12)/r12",
            "eri_hartree": eri,
            "derivative_first_center_ha_bohr": gradient.tolist(),
        },
        indent=2,
    )
)
