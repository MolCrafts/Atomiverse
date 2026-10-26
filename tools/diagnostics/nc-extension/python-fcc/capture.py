#!/usr/bin/env python3
"""Actual serial QE adjudication of the existing Python FCC fixture."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
binary = Path('/tmp/atv-qe-7.5-build/bin/pw.x').resolve()
source = Path('/tmp/atv-qe-7.5-src')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
# These are the existing Python assertion windows; this script does not edit them.
metadata = {
    'bounds_unchanged': {'energy_ev': .001, 'max_force_ev_ang': .005, 'net_force_ev_ang': 1e-4},
    'fixture': 'Si fcc primitive a5.43Ang, R0=0,R1=(a/4+.08,a/4,a/4); published GTH-PBE-q4; Gamma;8/32Ry;8e,4bands,fixed restricted',
    'native_xc': {'LSDA': 'Slater+VWN5 (CPU xc_functional.cpp)', 'PBE': 'Slater/PBE-X+PW92/PBE-C'},
    'qe_total_force_projection': 'PW/src/forces.f90 subtracts sumfor/nat before printing total forces; raw contribution sum is also recorded',
    'binary': str(binary), 'binary_sha256': sha(binary),
    'environment_requested': {k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    'source_sha256': {p: sha(source/p) for p in ['PW/src/forces.f90', 'XClib/qe_funct_corr_lda_lsda.f90', 'upflib/gth.f90']},
    'potential_sha256': sha(root.parent/'Si_GTH_PBE_q4.gth'),
    'runs': [],
}
# All runs are serial, including paired translated energies. No native solver.
for name, xc, grid, shift in [
    ('lsda14', 'SLA VWN', 14, 0), ('lsda56', 'SLA VWN', 56, 0),
    ('pbe22', 'PBE', 22, 0), ('pbe44', 'PBE', 44, 0),
    ('pbe88', 'PBE', 88, 0),
    ('pbe22-xminus', 'PBE', 22, -.001), ('pbe22-xplus', 'PBE', 22, .001),
]:
    inp = f'''&CONTROL
 calculation='scf', prefix='{name}', pseudo_dir='../',
 outdir='./scratch-{name}', tprnfor=.true., tstress=.false., verbosity='high'
/
&SYSTEM
 ibrav=0, nat=2, ntyp=1, ecutwfc=8.0, ecutrho=32.0,
 nr1={grid}, nr2={grid}, nr3={grid}, nr1s={grid}, nr2s={grid}, nr3s={grid},
 input_dft='{xc}', occupations='fixed', nspin=1, nbnd=4,
 nosym=.true., noinv=.true.
/
&ELECTRONS
 conv_thr=1.0d-12, electron_maxstep=200, mixing_beta=0.3,
 diagonalization='david', diago_thr_init=1.0d-13
/
ATOMIC_SPECIES
Si 28.0855 Si_GTH_PBE_q4.gth
CELL_PARAMETERS angstrom
0 2.715 2.715
2.715 0 2.715
2.715 2.715 0
ATOMIC_POSITIONS angstrom
Si {shift:.12f} 0 0
Si {1.4375+shift:.12f} 1.3575 1.3575
K_POINTS gamma
'''
    (root/f'{name}.in').write_text(inp)
    cmd=[str(binary), '-in', f'{name}.in']
    with (root/f'{name}.out.txt').open('w') as out:
        code=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
    raw=(root/f'{name}.out.txt').read_text()
    total_block=raw.split('Forces acting on atoms (cartesian axes, Ry/au):')[-1].split('The non-local')[0]
    force_re=r'atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)'
    energy=re.findall(r'!\s+total energy\s*=\s*([-+\d.Ee]+)\s+Ry',raw)
    forces=[[float(x) for x in row] for row in re.findall(force_re,total_block)]
    if code or 'JOB DONE.' not in raw or not energy or len(forces)!=2 or 'convergence NOT achieved' in raw:
        raise SystemExit(f'{name} not converged/completed; raw retained; no accepted reference')
    contributions={}
    for match in re.finditer(r'The ([^\n]+)\n((?:\s*atom[^\n]*\n){2})',raw):
        values=re.findall(force_re,match.group(2))
        if len(values)==2:
            contributions[match.group(1).strip()]=[[float(x) for x in row] for row in values]
    raw_forces=[[sum(values[i][d] for values in contributions.values()) for d in range(3)] for i in range(2)]
    entry={
        'name':name,'command':cmd,'xc':xc,'grid':grid,'rigid_shift_x_ang':shift,
        'engine':re.search(r'Program PWSCF[^\n]+',raw).group(0).strip(),
        'qe_source_commit':re.search(r'Last git commit:\s*([^\n]+)',raw).group(1).strip(),
        'scf_iterations':int(re.search(r'convergence has been achieved in\s+(\d+) iterations',raw).group(1)),
        'energy_ry':float(energy[-1]),'total_forces_ry_bohr':forces,
        'preprojection_contributions_ry_bohr':contributions,
        'preprojection_forces_ry_bohr':raw_forces,
        'preprojection_net_force_ry_bohr':[sum(row[d] for row in raw_forces) for d in range(3)],
        'printed_process_wall':re.search(r'PWSCF\s*:[^\n]+',raw).group(0).strip(),
        'files_sha256':{f'{name}.{suffix}':sha(root/f'{name}.{suffix}') for suffix in ['in','out.txt']},
    }
    metadata['runs'].append(entry)
    metadata['captured_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata['script_sha256']=sha(root/'capture.py')
    (root/'reference.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(name,entry['energy_ry'],forces,entry['preprojection_net_force_ry_bohr'],entry['printed_process_wall'],flush=True)
