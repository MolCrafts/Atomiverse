#!/usr/bin/env python3
"""Compare Atomiverse CPU / CUDA and LAMMPS single points term by term.

usage: compare.py <run_dir>

<run_dir> holds cpu.<term>.txt and cuda.<term>.txt (the regression
harnesses), log.<term> (LAMMPS, with the ATV_RESULT / ATV_VIRIAL lines of
in.lmp), forces.<term>.dump, and the ref_*.txt sums of exact_terms.py.
The real-space Coulomb part of each engine is `coul - kspace` (energy,
forces and virial). Metrics:
  rel dE   |E_a - E_b| / |E_b|
  F max    max_i |F_a,i - F_b,i| / max_i |F_b,i|     (vector norms)
  F rms    ||F_a - F_b||_2 / ||F_b||_2
  W        max_k |W_a,k - W_b,k| / max_k |W_b,k|     (six components)
Pure Python, no numpy.
"""
import math
import os
import sys

TERMS = ['fene', 'angle', 'lj', 'r4', 'coul', 'kspace', 'real', 'total']
SUMMED = ['fene', 'angle', 'lj', 'r4', 'coul']  # sum to total
NKTV2P = 1.6021765e6  # LAMMPS metal: bar per (eV / A^3)


def read_atv(path):
    out = {'forces': {}}
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if tok[0] == 'virial':
                out['virial'] = [float(x) for x in tok[1:7]]
            elif tok[0].startswith('energy'):
                out[tok[0]] = float(tok[1])
            else:
                out['forces'][int(tok[0])] = [float(x) for x in tok[1:4]]
    return out


def read_lammps(run_dir, term):
    out = {'forces': {}}
    with open(os.path.join(run_dir, 'log.' + term)) as fh:
        for line in fh:
            if line.startswith('ATV_RESULT'):
                tok = line.split()
                out.update({k: float(v) for k, v in zip(tok[3::2], tok[4::2])})
            elif line.startswith('ATV_VIRIAL'):
                out['pvirial'] = [float(x) for x in line.split()[3:9]]
    out['virial'] = [x * out['vol'] / NKTV2P for x in out['pvirial']]
    with open(os.path.join(run_dir, 'forces.%s.dump' % term)) as fh:
        lines = fh.read().splitlines()
    start = lines.index('ITEM: ATOMS id fx fy fz') + 1
    for line in lines[start:]:
        tok = line.split()
        out['forces'][int(tok[0])] = [float(x) for x in tok[1:4]]
    out['energy'] = out['pe']
    return out


def difference(a, b):
    return {'energy': a['energy'] - b['energy'],
            'virial': [x - y for x, y in zip(a['virial'], b['virial'])],
            'forces': {i: [x - y for x, y in zip(a['forces'][i],
                                                 b['forces'][i])]
                       for i in a['forces']}}


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def force_metrics(fa, fb):
    ids = sorted(fb)
    num_max = max(math.dist(fa[i], fb[i]) for i in ids)
    den_max = max(math.hypot(*fb[i]) for i in ids)
    num2 = sum(math.dist(fa[i], fb[i]) ** 2 for i in ids)
    den2 = sum(sum(x * x for x in fb[i]) for i in ids)
    return num_max / den_max, math.sqrt(num2 / den2)


def virial_metric(wa, wb):
    return (max(abs(a - b) for a, b in zip(wa, wb))
            / max(abs(b) for b in wb))


HEAD = '%-7s %24s %24s %9s %9s %9s %9s' % (
    'term', 'E_a [eV]', 'E_b [eV]', 'rel dE', 'F max', 'F rms', 'W')


def row(t, a, b):
    fmax, frms = force_metrics(a['forces'], b['forces'])
    print('%-7s %24.16e %24.16e %9.2e %9.2e %9.2e %9.2e' % (
        t, a['energy'], b['energy'], rel(a['energy'], b['energy']), fmax,
        frms, virial_metric(a['virial'], b['virial'])))


def table(name, a, b):
    print('\n' + name)
    print(HEAD)
    for t in TERMS:
        if t in a and t in b:
            row(t, a[t], b[t])


def main():
    run_dir = sys.argv[1]
    data = {}
    for side in ('cpu', 'cuda'):
        data[side] = {}
        for t in TERMS:
            p = os.path.join(run_dir, '%s.%s.txt' % (side, t))
            if os.path.exists(p):
                data[side][t] = read_atv(p)
    data['lammps'] = {}
    for t in TERMS:
        if os.path.exists(os.path.join(run_dir, 'log.' + t)):
            data['lammps'][t] = read_lammps(run_dir, t)
    for d in data.values():
        if 'coul' in d and 'kspace' in d:
            d['real'] = difference(d['coul'], d['kspace'])
    refs = {}
    for name in ('exact.real', 'lammps_as.real', 'exact.r4'):
        p = os.path.join(run_dir, 'ref_%s.txt' % name)
        if os.path.exists(p):
            refs[name] = read_atv(p)

    table('Atomiverse CPU (a) vs LAMMPS (b)', data['cpu'], data['lammps'])
    table('Atomiverse CUDA (a) vs LAMMPS (b)', data['cuda'], data['lammps'])
    table('Atomiverse CUDA (a) vs CPU (b)', data['cuda'], data['cpu'])

    print('\nEngine (a) vs direct pair sum (b, exact_terms.py)')
    print(HEAD.replace('term   ', 'a / b  '))
    pairs = [('cpu', 'real', 'exact.real'), ('cuda', 'real', 'exact.real'),
             ('lammps', 'real', 'exact.real'),
             ('lammps', 'real', 'lammps_as.real'),
             ('cpu', 'r4', 'exact.r4'), ('cuda', 'r4', 'exact.r4'),
             ('lammps', 'r4', 'exact.r4')]
    for side, t, ref in pairs:
        if t in data[side] and ref in refs:
            print('%s / %s' % (side, ref))
            row(t, data[side][t], refs[ref])

    for side in ('cpu', 'cuda', 'lammps'):
        d = data[side]
        if all(t in d for t in SUMMED + ['total']):
            s = sum(d[t]['energy'] for t in SUMMED)
            print('\n%s: sum of terms %.16e, total %.16e, rel %.2e' % (
                side, s, d['total']['energy'], rel(s, d['total']['energy'])))


if __name__ == '__main__':
    main()
