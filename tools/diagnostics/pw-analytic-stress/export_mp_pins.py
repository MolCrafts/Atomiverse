#!/usr/bin/env python3
"""Rebuild the one matched MP1 pin from immutable actual QE7.6 input/XML/stdout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
p.add_argument('--raw-root', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--check', action='store_true')
a = p.parse_args()
d = a.raw_root or a.root/'results/2026-10-28-mp-gamma-pbe/private-candidate'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(d/'ready.json') == 'b4e68a9f3d9a450bb68f8e2ab84259d4788c24c38d9a14dbaf309bf48fe22b33'
m = json.loads((d/'ready.json').read_text())
r = json.loads((d/'qe-reference.json').read_text())
required = ['qe-reference.json', 'qe/input.in', 'qe/qe.out', 'qe/exit-code.txt',
            'qe/H_GTH_PBE_q1.gth', 'qe/binary.sha256',
            'qe/scratch/h3.save/data-file-schema.xml']
for f in required:
    assert sha(d/f) == m['files_sha256'][f], ('raw hash', f)
assert (d/'qe/exit-code.txt').read_text().strip() == '0'
raw = (d/'qe/qe.out').read_text()
assert 'Program PWSCF v.7.6' in raw
assert 'JOB DONE.' in raw and 'convergence has been achieved' in raw
assert 'convergence NOT achieved' not in raw
inp = (d/'qe/input.in').read_text()
for token in ["input_dft='PBE'", "smearing='methfessel-paxton'", 'degauss=0.04',
              'ecutwfc=6.0', 'nspin=2', 'nbnd=5', 'electron_maxstep=250',
              'mixing_beta=0.35', 'K_POINTS gamma']:
    assert token in inp, ('input contract', token)
assert 'tot_magnetization' not in inp
assert sha(d/'qe/H_GTH_PBE_q1.gth') == 'a08012117c66d3f725ac5e672e1a061a22c41c4d16aa4d41b6863e1c9e634916'
o = ET.parse(d/'qe/scratch/h3.save/data-file-schema.xml').getroot().find('output')
assert o.findtext('band_structure/lsda') == 'true'
assert int(o.findtext('band_structure/nks')) == 1
for grid in ['fft_grid', 'fft_smooth']:
    node = o.find('basis_set/'+grid)
    assert [int(node.attrib[k]) for k in ['nr1', 'nr2', 'nr3']] == [48,42,42]
f = list(map(float, o.findtext('forces').split()))
s = list(map(float, o.findtext('stress').split()))
assert len(f) == 9 and len(s) == 9
assert max(abs(s[i*3+j]-s[j*3+i]) for i in range(3) for j in range(3)) < 1e-14
E = float(o.findtext('total_energy/etot'))
Ne = float(o.findtext('band_structure/nelec'))
M = float(o.findtext('magnetization/total'))
S_Ry = 2*float(o.findtext('total_energy/demet'))
occ = list(map(float, o.findtext('band_structure/ks_energies/occupations').split()))
assert len(occ) == 10 and min(occ) < -1e-4 and max(occ) > 1.0001
assert abs(sum(occ)-Ne) < 1e-8 and abs(sum(occ[:5])-sum(occ[5:])-M) < 1e-8
assert Ne == 3 and abs(M) > .05 and S_Ry > 1e-6
six = [s[i] for i in [0,4,8,1,2,5]]
assert E == r['E_Ha'] and M == r['M'] and S_Ry == r['minusTS_Ry']
assert [f[i:i+3] for i in range(0,9,3)] == r['fullF_HaBohr'] and six == r['sigma_HaBohr3']
# Raw text carries Ry units; XML carries Hartree units. Keep the printed cross-check.
assert abs(float(re.findall(r'!\s+total energy\s+=\s+([-+0-9.Ee]+)', raw)[-1])/2-E) < 3e-9
force_rows = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force\s+=\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)', raw)
assert len(force_rows) >= 3
force_rows = force_rows[:3]  # first block is total; later blocks are component detail
assert max(abs(float(v)/2-f[i]) for i,v in enumerate(sum((list(x) for x in force_rows), []))) < 3e-9
stress_tail = raw.split('total   stress')[-1].splitlines()[1:4]
printed_s = [float(v) for line in stress_tail for v in line.split()[:3]]
assert len(printed_s) == 9 and max(abs(v/2-w) for v,w in zip(printed_s,s)) < 3e-9
fmt = lambda v: format(v, '.17g')
forces = '{{'+','.join('{'+','.join(fmt(x) for x in f[i:i+3])+'}' for i in range(0,9,3))+'}}'
text = ('#pragma once\n#include "multi_k_fd_pins.h"\nnamespace pw_mp_qe {\n'
        'inline constexpr pw_multi_k_fd_qe::Pin pin = {\n'
        '"h3-gamma-uks-pbe-mp1-004",pw_multi_k_fd_qe::Model::PBE,true,3,{48,42,42},'
        + ','.join(map(fmt,[Ne,M,S_Ry,E])) + ','+forces+',{'+','.join(map(fmt,six))+'}};\n}\n')
def tokens(t):
    t = re.sub(r'//[^\n]*|/\*.*?\*/', '', t, flags=re.S)
    return re.findall(r'"(?:\\.|[^"\\])*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|::|\S',t)
if a.check:
    assert tokens(text) == tokens((a.root/'mp_ensemble_pins.h').read_text()), 'literal token mismatch'
if a.output:
    a.output.write_text(text)
print('PASS actualQE MP1 raw hashes and literal pin tokens: 1 matched PBE Gamma UKS reference')
