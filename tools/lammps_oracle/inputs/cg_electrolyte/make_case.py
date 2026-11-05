#!/usr/bin/env python3
"""Generate the coarse-grained polymer-electrolyte single-point case.

Writes, into the directory given as argv[1] (default: this directory):

  system.txt   the Atomiverse harness input (parameters, atoms, topology),
  data.lmp     the LAMMPS data file of the same configuration,
  in.lmp       the LAMMPS input (one `-var term <name>` per energy term),
  lj_rows*.lmp every LJ pair_coeff row (plain and hybrid/overlay forms),
  r4.table     the LAMMPS `pair_style table` file of the r^-4 L-E term.

Model (reduced units sigma0 = 1 A, m = 1 amu, eps0 = k_B * 300 K):
types B E C A T L (1..6), sigma = 1.00 1.00 1.10 1.15 1.20 0.55,
eps_ii = 1 1 1.2 1.2 1 1, Lorentz-Berthelot with eps_LT = 1.0 and
eps_LA = 1.2 overrides, every LJ pair cut and shifted at 2.5 sigma_ij;
charges z = +1 (C, L), -1 (A, T) scaled by q0 = sqrt(eps0 * lB* * sigma0 /
14.399645) e with lB* = 3; FENE K = 30 eps0/sigma0^2, R0 = 1.5 sigma0, no
WCA; cosine angle K = 2 eps0 on B-B-B; L-E r^-4 term
U = -S[(s/r)^4 - (s/rc)^4 + 4 (s/rc)^4 (r/rc - 1)], S = 4 eps0, s = 0.775,
rc = 2.5 s; special_bonds lj/coul 1 1 1; PPPM.

Pure Python (no numpy) and deterministic: random.Random(SEED).
"""
import math
import os
import random
import sys

SEED = 20261029
L = 9.0                       # cubic box edge [A]
KB = 8.617333262e-5           # Boltzmann constant [eV/K] (CODATA 2018)
EPS0 = KB * 300.0             # energy unit [eV]
QQR2E = 14.399645             # LAMMPS metal / ATV units::coulomb [eV A / e^2]
LB_STAR = 3.0                 # reduced Bjerrum length
Q0 = math.sqrt(EPS0 * LB_STAR * 1.0 / QQR2E)  # charge unit [e]

NAMES = ['B', 'E', 'C', 'A', 'T', 'L']          # LAMMPS / ATV types 1..6
SIGMA = {1: 1.00, 2: 1.00, 3: 1.10, 4: 1.15, 5: 1.20, 6: 0.55}
EPS = {1: 1.0, 2: 1.0, 3: 1.2, 4: 1.2, 5: 1.0, 6: 1.0}  # in eps0
OVERRIDE = {(5, 6): 1.0, (4, 6): 1.2}           # eps_LT, eps_LA in eps0
Z = {1: 0, 2: 0, 3: +1, 4: -1, 5: -1, 6: +1}
CUT_FACTOR = 2.5
RC_COUL = 3.5                 # Coulomb cutoff [A]
CUT_LJ_GLOBAL = 3.0           # LAMMPS global LJ cutoff (every row has its own)
ALPHA = 1.1                   # Ewald alpha [1/A]
MESH = 32                     # PPPM mesh per axis (spacing L/32)
# PPPM assignment order (CG_ORDER=5 / 7 for the odd-order variants). Both
# engines spread on LAMMPS's centred stencil nx + nlower .. nx + nupper.
ORDER = int(os.environ.get('CG_ORDER', 6))
FENE_K, FENE_R0 = 30.0, 1.5   # [eps0/sigma0^2], [sigma0]
COS_K = 2.0                   # [eps0]
R4_S, R4_SIGMA = 4.0, 0.775   # [eps0], [A]
R4_RC = 2.5 * R4_SIGMA
R4_INNER = 0.3                # table inner radius [A]
R4_N = int(os.environ.get('CG_R4_N', 200000))  # table points (RSQ spacing)
BOND = 0.97                   # construction bond length [A]
DMIN = 0.85                   # minimum nonbonded distance / sigma_ij

N_CHAINS = 6
# Per chain: 8 PEGM (B-E-E-E), 2 METAC (B-C), 2 MTFSI (B-A) monomers.
MONOMERS = ['PEGM'] * 8 + ['METAC'] * 2 + ['MTFSI'] * 2
N_FREE = 60                   # free L and free T each


def sij(a, b):
    return 0.5 * (SIGMA[a] + SIGMA[b])


def eij(a, b):
    key = (min(a, b), max(a, b))
    if key in OVERRIDE:
        return OVERRIDE[key]
    return math.sqrt(EPS[a] * EPS[b])


def build(rng):
    types, pos, mol, bonds, angles = [], [], [], [], []

    def dist(p, q):
        d = [p[k] - q[k] for k in range(3)]
        d = [x - L * round(x / L) for x in d]
        return math.sqrt(sum(x * x for x in d))

    def fits(p, t, skip):
        for k, q in enumerate(pos):
            if k in skip:
                continue
            if dist(p, q) < DMIN * sij(t, types[k]):
                return False
        return True

    def wrap(p):
        return [x - L * math.floor(x / L) for x in p]

    def unit():
        while True:
            v = [rng.uniform(-1, 1) for _ in range(3)]
            n = math.sqrt(sum(x * x for x in v))
            if 0.1 < n <= 1.0:
                return [x / n for x in v]

    def grow(anchor, t, prev_dir=None, tries=400):
        for _ in range(tries):
            u = unit()
            # Bond angle at the anchor > ~75 degrees from the previous bond.
            if prev_dir is not None and sum(
                    a * b for a, b in zip(u, prev_dir)) > -0.25:
                continue
            p = wrap([pos[anchor][k] + BOND * u[k] for k in range(3)])
            if fits(p, t, {anchor}):
                return p, u
        return None, None

    def add(t, p, m):
        types.append(t)
        pos.append(p)
        mol.append(m)
        return len(types) - 1

    for c in range(N_CHAINS):
        while True:
            start = len(types)
            n_bonds, n_angles = len(bonds), len(angles)
            seq = MONOMERS[:]
            rng.shuffle(seq)
            ok = True
            backbone, last_dir = [], None
            for m in seq:
                if not backbone:
                    for _ in range(1000):
                        p = [rng.uniform(0, L) for _ in range(3)]
                        if fits(p, 1, set()):
                            break
                    else:
                        ok = False
                        break
                    u = None
                else:
                    # backward direction of the previous backbone bond
                    back = None if last_dir is None else [-x for x in last_dir]
                    p, u = grow(backbone[-1], 1, back)
                    if p is None:
                        ok = False
                        break
                b = add(1, p, c + 1)
                if backbone:
                    bonds.append((backbone[-1], b))
                    if len(backbone) >= 2:
                        angles.append((backbone[-2], backbone[-1], b))
                backbone.append(b)
                last_dir = u
                side = {'PEGM': [2, 2, 2], 'METAC': [3], 'MTFSI': [4]}[m]
                anchor, d = b, None
                for t in side:
                    p, d = grow(anchor, t, None if d is None else [-x for x in d])
                    if p is None:
                        ok = False
                        break
                    s = add(t, p, c + 1)
                    bonds.append((anchor, s))
                    anchor = s
                if not ok:
                    break
            if ok:
                break
            del types[start:], pos[start:], mol[start:]
            del bonds[n_bonds:], angles[n_angles:]
    m = N_CHAINS
    for t in [6] * N_FREE + [5] * N_FREE:
        for _ in range(100000):
            p = [rng.uniform(0, L) for _ in range(3)]
            if fits(p, t, set()):
                break
        else:
            raise RuntimeError('free-ion insertion failed')
        m += 1
        add(t, p, m)
    return types, pos, mol, bonds, angles


def r4_energy_force(r):
    s4 = R4_SIGMA ** 4
    a = R4_S * EPS0 * s4
    e = -a / r ** 4 - 4.0 * a * r / R4_RC ** 5 + 5.0 * a / R4_RC ** 4
    f = -4.0 * a * (1.0 / r ** 5 - 1.0 / R4_RC ** 5)  # -dU/dr
    return e, f


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.abspath(__file__))
    rng = random.Random(SEED)
    types, pos, mol, bonds, angles = build(rng)
    n = len(types)
    q = [Z[t] * Q0 for t in types]
    assert abs(sum(Z[t] for t in types)) == 0

    with open(os.path.join(out, 'system.txt'), 'w') as fh:
        fh.write('# cg-electrolyte single point, make_case.py seed %d\n' % SEED)
        fh.write('box %.17g\n' % L)
        fh.write('eps0 %.17g\n' % EPS0)
        fh.write('q0 %.17g\n' % Q0)
        fh.write('alpha %.17g\n' % ALPHA)
        fh.write('rc_coul %.17g\n' % RC_COUL)
        fh.write('cut_factor %.17g\n' % CUT_FACTOR)
        fh.write('mesh %d\n' % MESH)
        fh.write('order %d\n' % ORDER)
        fh.write('fene %.17g %.17g\n' % (FENE_K, FENE_R0))
        fh.write('cosine %.17g\n' % COS_K)
        fh.write('r4 2 6 %.17g %.17g %.17g\n' % (R4_S, R4_SIGMA, R4_RC))
        fh.write('ntypes %d\n' % len(NAMES))
        for t in range(1, 7):
            fh.write('type %d %.17g %.17g\n' % (t, SIGMA[t], EPS[t]))
        fh.write('noverride %d\n' % len(OVERRIDE))
        for (a, b), e in sorted(OVERRIDE.items()):
            fh.write('override %d %d %.17g\n' % (a, b, e))
        fh.write('natoms %d\n' % n)
        for i in range(n):
            fh.write('%d %d %.17g %.17g %.17g\n' % (types[i], Z[types[i]],
                                                    *pos[i]))
        fh.write('nbonds %d\n' % len(bonds))
        for a, b in bonds:
            fh.write('%d %d\n' % (a, b))
        fh.write('nangles %d\n' % len(angles))
        for a, b, c in angles:
            fh.write('%d %d %d\n' % (a, b, c))

    with open(os.path.join(out, 'data.lmp'), 'w') as fh:
        fh.write('cg-electrolyte (make_case.py seed %d)\n\n' % SEED)
        fh.write('%d atoms\n%d bonds\n%d angles\n\n' % (n, len(bonds),
                                                      len(angles)))
        fh.write('6 atom types\n1 bond types\n1 angle types\n\n')
        fh.write('0 %.17g xlo xhi\n0 %.17g ylo yhi\n0 %.17g zlo zhi\n\n'
                 % (L, L, L))
        fh.write('Masses\n\n')
        for t in range(1, 7):
            fh.write('%d 1.0 # %s\n' % (t, NAMES[t - 1]))
        fh.write('\nAtoms # full\n\n')
        for i in range(n):
            fh.write('%d %d %d %.17g %.17g %.17g %.17g\n'
                     % (i + 1, mol[i], types[i], q[i], *pos[i]))
        fh.write('\nBonds\n\n')
        for k, (a, b) in enumerate(bonds):
            fh.write('%d 1 %d %d\n' % (k + 1, a + 1, b + 1))
        fh.write('\nAngles\n\n')
        for k, (a, b, c) in enumerate(angles):
            fh.write('%d 1 %d %d %d\n' % (k + 1, a + 1, b + 1, c + 1))

    # Every LJ row explicit (no LAMMPS mixing): eps, sigma, cut 2.5 sigma_ij.
    rows = []
    for a in range(1, 7):
        for b in range(a, 7):
            s = sij(a, b)
            rows.append((a, b, eij(a, b) * EPS0, s, CUT_FACTOR * s))

    with open(os.path.join(out, 'r4.table'), 'w') as fh:
        fh.write('# L-E force-shifted r^-4 term, make_case.py\n\nR4\n')
        fh.write('N %d RSQ %.17g %.17g\n\n' % (R4_N, R4_INNER, R4_RC))
        lo, hi = R4_INNER ** 2, R4_RC ** 2
        for i in range(R4_N):
            r = math.sqrt(lo + (hi - lo) * i / (R4_N - 1))
            e, f = r4_energy_force(r)
            fh.write('%d %.17g %.17g %.17g\n' % (i + 1, r, e, f))

    for name, style in (('lj_rows.lmp', ''),
                        ('lj_rows_hybrid.lmp', ' lj/cut/coul/long')):
        with open(os.path.join(out, name), 'w') as fh:
            for a, b, e, s, c in rows:
                fh.write('pair_coeff %d %d%s %.17g %.17g %.17g\n'
                         % (a, b, style, e, s, c))
    with open(os.path.join(out, 'in.lmp'), 'w') as fh:
        fh.write(IN_TEMPLATE.format(
            fene_k=FENE_K * EPS0, fene_r0=FENE_R0, cos_k=COS_K * EPS0,
            cut_lj=CUT_LJ_GLOBAL, rc_coul=RC_COUL, alpha=ALPHA, mesh=MESH,
            order=ORDER, r4n=R4_N, r4rc=R4_RC))
    print('atoms %d bonds %d angles %d q0 %.17g eps0 %.17g' %
          (n, len(bonds), len(angles), Q0, EPS0))


IN_TEMPLATE = r'''# cg-electrolyte single point (generated by make_case.py).
# lmp -in in.lmp -var term <fene|angle|lj|r4|coul|kspace|total>
units           metal
atom_style      full
boundary        p p p
read_data       data.lmp
special_bonds   lj/coul 1.0 1.0 1.0
neighbor        0.5 bin
neigh_modify    every 1 delay 0 check no one 10000

if "${{term}} == fene" then &
  "bond_style fene" "bond_coeff 1 {fene_k:.17g} {fene_r0:.17g} 0.0 1.0" &
  "angle_style zero" "angle_coeff *" &
  "pair_style zero {rc_coul:.17g}" "pair_coeff * *"
if "${{term}} == angle" then &
  "bond_style zero" "bond_coeff *" &
  "angle_style cosine" "angle_coeff 1 {cos_k:.17g}" &
  "pair_style zero {rc_coul:.17g}" "pair_coeff * *"
if "${{term}} == lj" then &
  "bond_style zero" "bond_coeff *" "angle_style zero" "angle_coeff *" &
  "pair_style lj/cut {cut_lj:.17g}" "pair_modify shift yes" &
  "include lj_rows.lmp"
if "${{term}} == r4" then &
  "bond_style zero" "bond_coeff *" "angle_style zero" "angle_coeff *" &
  "pair_style hybrid zero {rc_coul:.17g} table linear {r4n}" &
  "pair_coeff * * zero" "pair_coeff 2 6 table r4.table R4 {r4rc:.17g}"
if "${{term}} == coul" then &
  "bond_style zero" "bond_coeff *" "angle_style zero" "angle_coeff *" &
  "pair_style coul/long {rc_coul:.17g}" "pair_coeff * *" &
  "pair_modify table 0" &
  "kspace_style pppm 1.0e-8" &
  "kspace_modify gewald {alpha:.17g} mesh {mesh} {mesh} {mesh} order {order}"
if "${{term}} == kspace" then &
  "bond_style zero" "bond_coeff *" "angle_style zero" "angle_coeff *" &
  "pair_style coul/long 0.05" "pair_coeff * *" &
  "pair_modify table 0" &
  "kspace_style pppm 1.0e-8" &
  "kspace_modify gewald {alpha:.17g} mesh {mesh} {mesh} {mesh} order {order}"
if "${{term}} == total" then &
  "bond_style fene" "bond_coeff 1 {fene_k:.17g} {fene_r0:.17g} 0.0 1.0" &
  "angle_style cosine" "angle_coeff 1 {cos_k:.17g}" &
  "pair_style hybrid/overlay lj/cut/coul/long {cut_lj:.17g} {rc_coul:.17g} table linear {r4n}" &
  "include lj_rows_hybrid.lmp" &
  "pair_coeff 2 6 table r4.table R4 {r4rc:.17g}" &
  "pair_modify pair lj/cut/coul/long shift yes table 0" &
  "kspace_style pppm 1.0e-8" &
  "kspace_modify gewald {alpha:.17g} mesh {mesh} {mesh} {mesh} order {order}"

compute         vir all pressure NULL virial
thermo_style    custom step pe ebond eangle evdwl ecoul elong vol
thermo_modify   norm no
run             0
print           "ATV_RESULT term ${{term}} pe $(pe:%.17g) ebond $(ebond:%.17g) eangle $(eangle:%.17g) evdwl $(evdwl:%.17g) ecoul $(ecoul:%.17g) elong $(elong:%.17g) vol $(vol:%.17g)"
print           "ATV_VIRIAL term ${{term}} $(c_vir[1]:%.17g) $(c_vir[2]:%.17g) $(c_vir[3]:%.17g) $(c_vir[4]:%.17g) $(c_vir[5]:%.17g) $(c_vir[6]:%.17g)"
write_dump      all custom forces.${{term}}.dump id fx fy fz modify sort id format float %.17g
'''

if __name__ == '__main__':
    main()
