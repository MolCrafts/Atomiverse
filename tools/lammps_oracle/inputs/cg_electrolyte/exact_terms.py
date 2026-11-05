#!/usr/bin/env python3
"""Direct pair sums of the two terms whose engines approximate a function.

usage: exact_terms.py <run_dir>

Reads <run_dir>/system.txt and writes, in the Atomiverse harness format
(energy, six-component virial, per-tag forces):

  ref_exact.real.txt     real-space Ewald Coulomb with libm erfc,
  ref_lammps_as.real.txt the same sum with LAMMPS's `pair_modify table 0`
                         erfc (Abramowitz-Stegun 7.1.26 polynomial and the
                         truncated EWALD_F = 1.12837917, src/ewald_const.h),
  ref_exact.r4.txt       the L-E r^-4 term from its closed form.

An extra consistency check that attributes the measured real-space and r^-4
differences to each engine's function approximation; it is not the oracle
(LAMMPS is). Pure Python; the box exceeds twice the cutoff, so one minimum
image per pair suffices.
"""
import math
import os
import sys

QQR2E = 14.399645
EWALD_P = 0.3275911
A1, A2, A3, A4, A5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027,
                      1.061405429)
EWALD_F_LAMMPS = 1.12837917
EWALD_F = 2.0 / math.sqrt(math.pi)


def read(path):
    s = {'atoms': []}
    with open(path) as fh:
        lines = [l.split() for l in fh if l.strip() and not l.startswith('#')]
    k = 0
    while k < len(lines):
        tok = lines[k]
        key = tok[0]
        if key == 'natoms':
            n = int(tok[1])
            s['atoms'] = [(int(t[0]), int(t[1]), float(t[2]), float(t[3]),
                           float(t[4])) for t in lines[k + 1:k + 1 + n]]
            k += n + 1
            continue
        if key in ('nbonds', 'nangles'):
            k += int(tok[1]) + 1
            continue
        s[key] = tok[1:]
        k += 1
    return s


def pair_sum(s, kernel, cut, select=None):
    L = float(s['box'][0])
    atoms = s['atoms']
    n = len(atoms)
    f = [[0.0, 0.0, 0.0] for _ in range(n)]
    w = [0.0] * 6
    e_tot = 0.0
    for i in range(n):
        ti, zi, xi, yi, wi = atoms[i]
        for j in range(i + 1, n):
            tj, zj, xj, yj, wj = atoms[j]
            if select and not select(ti, tj, zi, zj):
                continue
            d = [xi - xj, yi - yj, wi - wj]
            d = [x - L * round(x / L) for x in d]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if r2 >= cut * cut:
                continue
            e, fpair = kernel(i, j, r2)
            e_tot += e
            for a in range(3):
                f[i][a] += fpair * d[a]
                f[j][a] -= fpair * d[a]
            w[0] += fpair * d[0] * d[0]
            w[1] += fpair * d[1] * d[1]
            w[2] += fpair * d[2] * d[2]
            w[3] += fpair * d[0] * d[1]
            w[4] += fpair * d[0] * d[2]
            w[5] += fpair * d[1] * d[2]
    return e_tot, w, f


def write(path, e, w, f):
    with open(path, 'w') as fh:
        fh.write('energy %.17g\nenergy_pair %.17g\nenergy_bond 0\n'
                 'energy_angle 0\n' % (e, e))
        fh.write('virial ' + ' '.join('%.17g' % x for x in w) + '\n')
        for i, fi in enumerate(f):
            fh.write('%d %.17g %.17g %.17g\n' % (i + 1, *fi))


def main():
    run = sys.argv[1]
    s = read(os.path.join(run, 'system.txt'))
    q0 = float(s['q0'][0])
    alpha = float(s['alpha'][0])
    rc = float(s['rc_coul'][0])
    q = [z * q0 for (_, z, _, _, _) in s['atoms']]
    charged = lambda ti, tj, zi, zj: zi != 0 and zj != 0

    def exact(i, j, r2):
        r = math.sqrt(r2)
        g = alpha * r
        qq = QQR2E * q[i] * q[j]
        ec = math.erfc(g)
        return qq * ec / r, qq * (ec + EWALD_F * g * math.exp(-g * g)) / (r2 * r)

    def lammps_as(i, j, r2):
        r = math.sqrt(r2)
        g = alpha * r
        ex = math.exp(-g * g)
        t = 1.0 / (1.0 + EWALD_P * g)
        ec = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * ex
        pref = QQR2E * q[i] * q[j] / r
        return pref * ec, pref * (ec + EWALD_F_LAMMPS * g * ex) / r2

    for name, k in (('exact', exact), ('lammps_as', lammps_as)):
        write(os.path.join(run, 'ref_%s.real.txt' % name),
              *pair_sum(s, k, rc, charged))

    eps0 = float(s['eps0'][0])
    ti, tj, S, sig, rc4 = s['r4']
    ti, tj = int(ti), int(tj)
    a = float(S) * eps0 * float(sig) ** 4
    rc4 = float(rc4)

    def r4(i, j, r2):
        r = math.sqrt(r2)
        e = -a / r2 ** 2 - 4.0 * a * r / rc4 ** 5 + 5.0 * a / rc4 ** 4
        fr = -4.0 * a * (1.0 / r ** 5 - 1.0 / rc4 ** 5)  # -dU/dr
        return e, fr / r

    pick = lambda a_, b_, zi, zj: {a_, b_} == {ti, tj}
    write(os.path.join(run, 'ref_exact.r4.txt'), *pair_sum(s, r4, rc4, pick))


if __name__ == '__main__':
    main()
