#!/usr/bin/env python3
"""Verify independent QE empty-beta raw data; export the one qualified LSDA pin."""
import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def tokens(s):
    s = re.sub(r'//[^\n]*|/\*.*?\*/', '', s, flags=re.S)
    return re.findall(r'"(?:\\.|[^"\\])*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|::|\S', s)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    rawroot = a.raw_root or a.root / 'empty-beta-rawrefs'
    if not rawroot.exists():
        rawroot = a.root / 'rawrefs'
    m = json.loads((rawroot / 'manifest.json').read_text())
    assert len(m['primary_cases']) == 1 and len(m['false_controls']) == 8
    for c in m['primary_cases'] + m['false_controls']:
        d = rawroot / c['raw_dir']
        for n, h in c['raw_files_sha256'].items():
            assert hashlib.sha256((d / n).read_bytes()).hexdigest() == h, (c['id'], n)
        raw = (d / 'qe.out').read_text()
        actual = 'convergence has been achieved' in raw and 'convergence NOT achieved' not in raw
        assert actual == c['QE_converged']
        assert (d / 'exit-code.txt').read_text().strip() == '0'
        assert 'Program PWSCF v.7.6' in raw and 'JOB DONE.' in raw
        if c in m['false_controls']:
            assert c['scientific_qualification_passed'] is False
    for n, h in m['historical_files_sha256'].items():
        assert hashlib.sha256((rawroot / 'historical-evidence' / n).read_bytes()).hexdigest() == h
    c = m['primary_cases'][0]
    assert c['id'] == 'h2-2A-triplet-lsda' and c['model'] == 'lsda'
    d = rawroot / c['raw_dir']
    inp = (d / 'input.in').read_text()
    for s in ["input_dft='SLA+VWN+NOGX+NOGC'", "occupations='fixed'", 'nspin=2', 'nbnd=2', 'tot_magnetization=2', 'K_POINTS gamma', 'electron_maxstep=250', 'mixing_beta=.35', 'ecutwfc=6']:
        assert s in inp, s
    assert 'degauss' not in inp and 'starting_magnetization' not in inp
    assert hashlib.sha256((d / 'H_GTH_PBE_q1.gth').read_bytes()).hexdigest() == 'a08012117c66d3f725ac5e672e1a061a22c41c4d16aa4d41b6863e1c9e634916'
    o = ET.parse(d / 'data-file-schema.xml').getroot().find('output')
    assert o.findtext('band_structure/lsda') == 'true'
    assert int(o.findtext('band_structure/nks')) == 1
    for g in ['fft_grid', 'fft_smooth']:
        grid = o.find('basis_set/' + g)
        assert [int(grid.attrib[k]) for k in ['nr1', 'nr2', 'nr3']] == c['matched_quad'] == [34, 30, 30]
    E = float(o.findtext('total_energy/etot'))
    f = list(map(float, o.findtext('forces').split()))
    s = list(map(float, o.findtext('stress').split()))
    Ne = float(o.findtext('band_structure/nelec'))
    M = float(o.findtext('magnetization/total'))
    S = 2 * float(o.findtext('total_energy/demet') or 0)
    ff = [f[i:i+3] for i in range(0, len(f), 3)]
    ss = [s[i] for i in [0, 4, 8, 1, 2, 5]]
    assert len(f) == 6 and len(s) == 9 and all(math.isfinite(v) for v in [E, Ne, M, S] + f + s)
    assert E == c['E'] and ff == c['f'] and ss == c['s'] and Ne == c['Ne'] and M == c['M'] and S == c['S_Ry'] == 0
    assert abs(Ne - 2) < 1e-8 and abs(M - 2) < 1e-8
    # Check full-precision XML against all printed total-force and total-stress rows.
    raw = (d / 'qe.out').read_text()
    energy = float(re.findall(r'!\s+total energy\s+=\s+([-\d.Ee+]+)', raw)[-1]) / 2
    assert abs(E - energy) < 3e-9
    block = raw[raw.rfind('Forces acting on atoms'):]
    printed_f = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force\s+=\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', block)
    assert len(printed_f) >= 2
    assert max(abs(float(v) / 2 - ref) for row, refs in zip(printed_f[:2], ff) for v, ref in zip(row, refs)) < 3e-9
    b = raw[raw.rfind('total   stress'):].splitlines()[1:4]
    printed_s = [float(v) / 2 for line in b for v in line.split()[:3]]
    assert len(printed_s) == 9 and max(abs(x-y) for x, y in zip(printed_s, s)) < 3e-9
    fmt = lambda v: format(v, '.17g')
    force = '{' + ','.join('{' + ','.join(map(fmt, row)) + '}' for row in ff + [[0, 0, 0]]) + '}'
    text = '\n'.join(['#pragma once', '#include "multi_k_fd_pins.h"', 'namespace empty_beta_qe {', 'inline constexpr pw_multi_k_fd_qe::Pin pin = {', '"h2-2A-triplet-lsda", pw_multi_k_fd_qe::Model::LSDA, true, 2, {34,30,30}, 2,2,0,', fmt(E) + ',{' + force + '},{' + ','.join(map(fmt, ss)) + '}};', '}']) + '\n'
    if a.check:
        assert tokens(text) == tokens((a.root / 'empty_beta_pins.h').read_text()), 'static pin tokens differ'
    if a.output:
        a.output.write_text(text)
    print('PASS actualQE empty-beta:1LSDAprimary/8preserved-false-controls; raw-hashes/fullF6sigma/exact-pin-tokens; tangent-only')


if __name__ == '__main__':
    main()
