#!/usr/bin/env python3
"""Run real QE once on si2.in and record its printed energy/forces.

The numbers pinned in examples/extensions/nc/pins.h come from this capture.
Inputs (si2.in, Si_GTH_PBE_q4.gth) are read from this directory; the QE run,
its raw stdout and reference.json go to OUT, which must not exist yet (pass
--parse-existing to re-read OUT/qe-raw.txt without running QE). No native
solve is involved.

    python capture.py /path/to/qe-7.5/bin/pw.x --out OUT
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

here = Path(__file__).resolve().parent
INPUTS = ('si2.in', 'Si_GTH_PBE_q4.gth')
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('binary', type=Path, help='QE pw.x executable (QE 7.5 for the current pin)')
parser.add_argument('--out', type=Path, required=True, help='work/output directory')
parser.add_argument('--parse-existing', action='store_true',
                    help='parse OUT/qe-raw.txt instead of running QE')
args = parser.parse_args()
binary = args.binary.resolve()
if not binary.is_file():
    raise SystemExit(f'pw.x not found: {binary}')
out = args.out.resolve()
if args.parse_existing:
    if not (out / 'qe-raw.txt').is_file():
        raise SystemExit(f'--parse-existing needs {out / "qe-raw.txt"}')
else:
    out.mkdir(parents=True, exist_ok=False)
    for name in INPUTS:
        shutil.copyfile(here / name, out / name)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
command = [str(binary), '-in', 'si2.in']
exit_code = 0
if not args.parse_existing:
    with (out / 'qe-raw.txt').open('w') as raw_file:
        exit_code = subprocess.run(command, cwd=out, env=env, stdout=raw_file,
                                   stderr=subprocess.STDOUT).returncode
raw = (out / 'qe-raw.txt').read_text()
energies = re.findall(r'!\s+total energy\s*=\s*([-+\d.Ee]+)\s+Ry', raw)
force_block = raw.split('Forces acting on atoms (cartesian axes, Ry/au):')[-1].split('The non-local')[0]
forces = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)', force_block)
if exit_code or 'JOB DONE.' not in raw or not energies or len(forces) != 2 or 'convergence NOT achieved' in raw:
    raise SystemExit(f'QE oracle incomplete; inspect {out / "qe-raw.txt"}; no pin written')
record = {
    'engine': re.search(r'Program PWSCF[^\n]+', raw).group(0).strip(),
    'command': command, 'cwd': str(out), 'exit_code': exit_code,
    'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'environment_requested': {k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    'binary': str(binary), 'binary_sha256': sha(binary),
    'files_sha256': {**{name: sha(here / name) for name in INPUTS},
                     'qe-raw.txt': sha(out / 'qe-raw.txt'), 'capture.py': sha(__file__)},
    'energy_ry': float(energies[-1]), 'forces_ry_per_bohr': [[float(x) for x in row] for row in forces],
    'bounds_fixed_before_native_run': {'energy_ha': 1e-5, 'each_force_ha_per_bohr': 1e-4},
    'hamiltonian': 'published Si GTH-PBE-q4, PBE, periodic cubic6 Ang, Si(2,2,2),(3.8,3.2,3), Gamma, ecut12/48 Ry, restricted FDsmear .02Ry6 bands',
    'grid': 'QE40^3 dense and smooth; native density base26^3 with PBE private40^3 quadrature; no stress qualification',
    'status': 'actual external pin only; no native validation claim',
    'qe_source_commit_reported': re.search(r'Last git commit:\s*([^\n]+)', raw).group(1).strip(),
    'printed_process_wall': re.search(r'PWSCF\s*:[^\n]+', raw).group(0).strip(),
    'metadata_extracted_from_existing_raw': args.parse_existing,
}
(out / 'reference.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
