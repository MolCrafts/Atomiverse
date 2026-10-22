#!/usr/bin/env python3
"""Matched GAFF PEO/OpenMM CUDA benchmark; run on a GPU batch allocation.

Each engine invocation is bounded by --budget (default 10 seconds). Installation
and compilation are outside that budget. No force-field arithmetic is used as
an oracle: energies and forces are independently evaluated by OpenMM itself.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from measure_resources import run_engine

EV_KCAL = 1 / 23.060547830619026
EV_KJ = EV_KCAL / 4.184


def supercell(rep, skew):
    """Integer rows M of a supercell lattice (A = sum_j M_ij a_j).

    ``skew`` keeps det(M) = rep**3 copies but tilts the cell:
    A = rep a, B = a + rep b, C = b + rep c. Any sublattice of the base
    lattice tiles the same infinite periodic system, so atom count, density
    and every bonded geometry are those of the orthorhombic replication.
    """
    if not skew:
        return [[rep, 0, 0], [0, rep, 0], [0, 0, rep]]
    if rep < 2:
        raise ValueError("--skew requires --replicate >= 2")
    return [[rep, 0, 0], [1, rep, 0], [0, 1, rep]]


def inverse3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    return [[(e*i-f*h)/det, (c*h-b*i)/det, (b*f-c*e)/det],
            [(f*g-d*i)/det, (a*i-c*g)/det, (c*d-a*f)/det],
            [(d*h-e*g)/det, (b*g-a*h)/det, (a*e-b*d)/det]], det


def load(data, ff, rep, skew=False):
    sections = {k: [] for k in ("Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals")}
    current = None
    bounds = []
    for line_number,line in enumerate(Path(data).read_text().splitlines()):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line_number == 0 and line.startswith('LAMMPS data file'):
            continue
        if line in sections:
            current = line
        elif line[0].isalpha():
            raise ValueError(f"unsupported LAMMPS section: {line}")
        elif line.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            lo, hi = map(float, line.split()[:2])
            bounds.append((lo, hi))
        elif line.endswith("xy xz yz"):
            raise ValueError("benchmark currently requires an orthorhombic input cell")
        elif current:
            sections[current].append(line.split())
    mass = {int(t[0]): float(t[1]) for t in sections["Masses"]}
    base = sorted(sections["Atoms"], key=lambda t: int(t[0]))
    if [int(t[0]) for t in base] != list(range(1, len(base)+1)):
        raise ValueError("contiguous atom IDs required")
    if len(bounds) != 3:
        raise ValueError("three box bounds required")
    lengths = [hi-lo for lo, hi in bounds]
    if rep < 1:
        raise ValueError("positive replication count required")
    M = supercell(rep, skew)
    Minv, det = inverse3(M)
    # Supercell rows in Angstrom (the base cell is orthorhombic).
    cell = [[M[i][j]*lengths[j] for j in range(3)] for i in range(3)]
    # Coset representatives t of base-lattice translations modulo the
    # supercell lattice: integer vectors with fractional t M^-1 in [0, 1).
    def frac(t):
        return [sum(t[i]*Minv[i][k] for i in range(3)) for k in range(3)]
    span = range(-rep, 2*rep+1)
    cosets = [(ix, iy, iz) for ix in span for iy in span for iz in span
              if all(-1e-12 <= f < 1-1e-12 for f in frac((ix, iy, iz)))]
    if len(cosets) != round(det):
        raise ValueError("supercell coset enumeration failed")
    coset_index = {t: n for n, t in enumerate(cosets)}
    def reduce(t):
        k = [math.floor(f+1e-12) for f in frac(t)]
        r = tuple(t[j]-sum(k[i]*M[i][j] for i in range(3)) for j in range(3))
        return coset_index[r]
    base_positions = [[(float(t[4+d])-bounds[d][0]) % lengths[d] for d in range(3)] for t in base]
    atoms, bonds, angles, dihedrals = [], [], [], []
    for t0 in cosets:
        for t, pos in zip(base, base_positions):
            typ = int(t[2])
            atoms.append([pos[d]+t0[d]*lengths[d] for d in range(3)]+[float(t[3]), typ-1, mass[typ]])
        for name, out in (("Bonds", bonds), ("Angles", angles), ("Dihedrals", dihedrals)):
            for t in sections[name]:
                # Follow each bonded path in minimum-image geometry.
                # A boundary-crossing endpoint belongs to the adjacent
                # translation, not the same one with a stretched bond.
                ids = [int(a)-1 for a in t[2:]]
                image = [0, 0, 0]
                mapped = []
                previous = ids[0]
                for atom in ids:
                    for d in range(3):
                        image[d] -= round((base_positions[atom][d]-base_positions[previous][d])/lengths[d])
                    target = tuple(t0[d]+image[d] for d in range(3))
                    mapped.append(reduce(target)*len(base)+atom)
                    previous = atom
                out.append(mapped+[int(t[1])-1])
    cell_inv, _ = inverse3(cell)
    def min_image(v):
        f = [sum(v[i]*cell_inv[i][k] for i in range(3)) for k in range(3)]
        f = [x-round(x) for x in f]
        return [sum(f[k]*cell[k][j] for k in range(3)) for j in range(3)]
    # Input geometry invariant: replication must preserve every consecutive
    # bonded displacement, not merely the atom count or total charge.
    for name, records in (("Bonds", bonds), ("Angles", angles), ("Dihedrals", dihedrals)):
        source = sections[name]
        for index, record in enumerate(records):
            ids = [int(a)-1 for a in source[index % len(source)][2:]]
            for edge in range(len(ids)-1):
                ref = [base_positions[ids[edge+1]][d]-base_positions[ids[edge]][d] for d in range(3)]
                ref = [ref[d]-lengths[d]*round(ref[d]/lengths[d]) for d in range(3)]
                new = min_image([atoms[record[edge+1]][d]-atoms[record[edge]][d] for d in range(3)])
                if max(abs(new[d]-ref[d]) for d in range(3)) > 1e-9:
                    raise ValueError(f"replication changed {name} geometry")
    coeff = {k: {} for k in ("pair", "bond", "angle", "dihedral")}
    for line in Path(ff).read_text().splitlines():
        t = line.split("#")[0].split()
        if not t or t[0] == "pair_modify":
            continue
        kind = t[0].removesuffix("_coeff")
        if kind not in coeff:
            raise ValueError(f"unsupported coefficient: {line}")
        idx = int(t[1])-1
        if kind == "pair":
            if t[1] != t[2]:
                raise ValueError("benchmark requires diagonal Lorentz–Berthelot LJ")
            val = [float(t[3])*EV_KCAL, float(t[4])]
        elif kind == "bond":
            val = [float(t[2])*EV_KCAL, float(t[3])]
        elif kind == "angle":
            val = [float(t[2])*EV_KCAL, math.radians(float(t[3]))]
        else:
            terms = int(t[2])
            if len(t) != 3+3*terms:
                raise ValueError("bad Fourier term count")
            val = [[float(t[3+3*j])*EV_KCAL, int(t[4+3*j]), math.radians(float(t[5+3*j]))]
                   for j in range(terms)]
        if idx in coeff[kind]:
            raise ValueError("duplicate coefficient")
        coeff[kind][idx] = val
    for kind, table in coeff.items():
        if sorted(table) != list(range(len(table))):
            raise ValueError(f"missing {kind} coefficient")
    return dict(atoms=atoms, bonds=bonds, angles=angles, dihedrals=dihedrals,
                box=[cell[d][d] for d in range(3)], cell=[v for row in cell for v in row],
                **{k:[v[t] for t in sorted(v)] for k,v in coeff.items()})


def polynomial(terms):
    # Exact identity for the phases/multiplicities in the PEO input. Reject
    # unsupported inputs instead of silently changing a Fourier force field.
    cheb = {1:[0,1,0,0,0], 2:[-1,0,2,0,0], 3:[0,-3,0,4,0], 4:[1,0,-8,0,8]}
    c = [0.0]*5
    for k,n,phase in terms:
        if n not in cheb or abs(math.sin(phase)) > 1e-12:
            raise ValueError("CUDA polynomial benchmark requires n=1..4 and phase=0 or pi")
        c[0] += k
        for j,v in enumerate(cheb[n]):
            c[j] += k*math.cos(phase)*v
    return c


def prepare(d, path, args):
    c = d['cell']
    rows = [[len(d['atoms']), *d['box'], c[3], c[6], c[7], len(d['pair']), len(d['bond']), len(d['angle']),
             len(d['dihedral']), len(d['bonds']), len(d['angles']), len(d['dihedrals']),
             args.alpha, args.cutoff, args.spacing, 5, args.dt, len(d.get('velocities', []))]]
    for name in ('pair','bond','angle'):
        rows.extend(d[name])
    rows.extend(polynomial(t) for t in d['dihedral'])
    for name in ('atoms','bonds','angles','dihedrals'):
        rows.extend(d[name])
    rows.extend(d.get('velocities', []))
    path.write_text('\n'.join(' '.join(map(str,r)) for r in rows)+'\n')


def openmm_system(d, args):
    """OpenMM System matching the ATV force field; returns (system, nonbonded)."""
    import openmm as mm
    import numpy as np
    from openmm import unit
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*[v/10 for v in d['cell'][3*i:3*i+3]])
                                         for i in range(3)])
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(nb.PME)
    nb.setCutoffDistance(args.cutoff/10)
    nb.setUseDispersionCorrection(False)
    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    def good(n):
        while True:
            m=n
            for p in (2,3,5,7):
                while m%p==0: m//=p
            if m==1: return n
            n+=1
    grid = [good(max(6,math.ceil(l/args.spacing))) for l in d['box']]
    nb.setPMEParameters(args.alpha*10,*grid)
    for x,y,z,q,t,mass in d['atoms']:
        system.addParticle(mass)
        eps,sig = d['pair'][t]
        nb.addParticle(q,sig/10,eps/EV_KJ)
    nb.createExceptionsFromBonds([(i,j) for i,j,t in d['bonds']],5/6,0.5)
    if args.term in ('all','nonbonded'): system.addForce(nb)
    b=mm.HarmonicBondForce();b.setUsesPeriodicBoundaryConditions(True)
    for i,j,t in d['bonds']:
        k,r=d['bond'][t];b.addBond(i,j,r/10,2*k/EV_KJ*100)
    if args.term in ('all','bond'): system.addForce(b)
    a=mm.HarmonicAngleForce();a.setUsesPeriodicBoundaryConditions(True)
    for i,j,k,t in d['angles']:
        force,theta=d['angle'][t];a.addAngle(i,j,k,theta,2*force/EV_KJ)
    if args.term in ('all','angle'): system.addForce(a)
    tor=mm.PeriodicTorsionForce();tor.setUsesPeriodicBoundaryConditions(True)
    for i,j,k,l,t in d['dihedrals']:
        for force,n,phase in d['dihedral'][t]:
            tor.addTorsion(i,j,k,l,n,phase,force/EV_KJ)
    if args.term in ('all','dihedral'): system.addForce(tor)
    return system, nb


def openmm_run(d, args, result):
    import openmm as mm
    import numpy as np
    from openmm import unit
    system, nb = openmm_system(d, args)
    integrator=mm.VerletIntegrator(args.dt/1000)
    platform=mm.Platform.getPlatformByName(args.platform)
    properties={'Precision':args.precision,'DeviceIndex':'0','UseCpuPme':'false'} if args.platform=='CUDA' else {}
    context=mm.Context(system,integrator,platform,properties)
    context.setPositions(np.asarray([a[:3] for a in d['atoms']])/10)
    context.setVelocities(np.zeros((len(d['atoms']),3)))
    state=context.getState(getEnergy=True,getForces=True)
    energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)*EV_KJ
    forces=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)*EV_KJ/10
    np.savetxt(result,forces,header=format(energy,'.17g'),comments='')
    # OpenMM Verlet stores leapfrog velocities at t-dt/2. Initialize that
    # half-step from the same zero full-step velocity used by ATV's velocity
    # Verlet, so positions and force evaluations follow the same trajectory.
    internal_forces=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    mass=np.asarray([a[5] for a in d['atoms']])
    context.setVelocities(-0.5*(args.dt/1000)*internal_forces/mass[:,None])
    integrator.step(args.warmup)
    context.getState(getEnergy=True)
    times=[]
    for sample in range(args.samples):
        start=time.perf_counter();integrator.step(args.steps)
        final_state=context.getState(getEnergy=True,getPositions=True,getVelocities=True,getForces=True)
        times.append((time.perf_counter()-start)*1000/args.steps)
    ms=statistics.median(times)
    np.savetxt(str(result)+'.final',final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    final_energy=final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)*EV_KJ
    if not math.isfinite(final_energy) or not np.isfinite(final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).all():
        raise RuntimeError('OpenMM final state is non-finite')
    return dict(engine='openmm',version=mm.__version__,platform=args.platform,N=len(d['atoms']),ms_per_step=ms,
                final_energy_ev=final_energy,sample_ms=times,
                precision=args.precision,pme=list(nb.getPMEParametersInContext(context))[1:] if args.term in ('all','nonbonded') else None)


MVV2E = 103.6427            # amu A^2/fs^2 -> eV (ATV units)
KB_EV = 8.617333262e-5      # eV/K


def thermal_velocities(d, temperature, seed):
    """Maxwell-Boltzmann velocities [A/fs] with zero total momentum."""
    import numpy as np
    rng = np.random.default_rng(seed)
    mass = np.asarray([a[5] for a in d['atoms']])
    v = rng.standard_normal((len(mass), 3))*np.sqrt(KB_EV*temperature/(MVV2E*mass))[:, None]
    v -= (mass[:, None]*v).sum(axis=0)/mass.sum()
    return v


def openmm_drift(d, args, dt, steps, every, velocities):
    """OpenMM NVE energy series (step, potential, kinetic) [eV]."""
    import openmm as mm
    import numpy as np
    from openmm import unit
    system, nb = openmm_system(d, args)
    integrator = mm.VerletIntegrator(dt/1000)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {'Precision': args.precision, 'DeviceIndex': '0', 'UseCpuPme': 'false'} if args.platform == 'CUDA' else {}
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(np.asarray([a[:3] for a in d['atoms']])/10)
    mass = np.asarray([a[5] for a in d['atoms']])
    state = context.getState(getForces=True)
    accel = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)/mass[:, None]
    # Leapfrog stores v(t - dt/2); start from the same full-step velocity.
    context.setVelocities(velocities*100 - 0.5*(dt/1000)*accel)
    series = []
    for done in range(0, steps+1, every):
        if done:
            integrator.step(every)
        st = context.getState(getEnergy=True)
        series.append((done, st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)*EV_KJ,
                       st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)*EV_KJ))
    return series


def drift_fit(series, dt, natoms):
    """Linear drift [eV/ps/atom] and residual RMS [eV/atom] of E_total."""
    import numpy as np
    a = np.asarray(series, dtype=float)
    t = a[:, 0]*dt/1000
    e = (a[:, 1]+a[:, 2])/natoms
    slope, intercept = np.polyfit(t, e, 1)
    return float(slope), float(np.sqrt(np.mean((e-(slope*t+intercept))**2)))


def run_drift(d, args, out):
    """NVE drift at dt and dt/2 for each seed, ATV against OpenMM."""
    import subprocess
    results = []
    for seed in [int(x) for x in args.seeds.split(',')]:
        v = thermal_velocities(d, args.temperature, seed)
        for dt in (args.dt, args.dt/2):
            steps = int(round(args.drift_time/dt))
            every = max(1, int(round(args.drift_every/dt)))
            case = dict(d, velocities=[[float(x) for x in row] for row in v])
            run_args = argparse.Namespace(**{**vars(args), 'dt': dt})
            tag = out/f'drift-s{seed}-dt{dt:g}'
            tag.mkdir(parents=True, exist_ok=True)
            prepare(case, tag/'input.txt', run_args)
            proc = subprocess.run([args.atv, str(tag/'input.txt'), str(tag/'atv.txt'), str(steps), '1', 'all', '1', str(every)],
                                  capture_output=True, text=True, timeout=args.budget)
            if proc.returncode:
                raise RuntimeError(f'ATV drift run failed:\n{proc.stdout}\n{proc.stderr}')
            atv = [tuple(map(float, line.split())) for line in (tag/'atv.txt.drift').read_text().splitlines()]
            omm = openmm_drift(case, run_args, dt, steps, every, v)
            (tag/'openmm.drift').write_text(''.join(f'{a} {b!r} {c!r}\n' for a, b, c in omm))
            sa, ra = drift_fit(atv, dt, len(d['atoms']))
            so, ro = drift_fit(omm, dt, len(d['atoms']))
            results.append(dict(seed=seed, dt_fs=dt, steps=steps, atv_drift_ev_ps_atom=sa, atv_rms_ev_atom=ra,
                                openmm_drift_ev_ps_atom=so, openmm_rms_ev_atom=ro))
            print(json.dumps(results[-1]), flush=True)
    # Drift slopes over a few ps are noise dominated (they do not shrink with
    # dt in either engine), so the seeds define the reference distribution:
    # the ATV mean slope must agree with OpenMM's within two pooled standard
    # errors, and the energy-fluctuation RMS within 20 %.
    import statistics
    comparison = {}
    for dt in (args.dt, args.dt/2):
        rows = [r for r in results if r['dt_fs'] == dt]
        a = [r['atv_drift_ev_ps_atom'] for r in rows]
        o = [r['openmm_drift_ev_ps_atom'] for r in rows]
        n = len(rows)
        se = math.sqrt((statistics.pvariance(a)+statistics.pvariance(o))/max(n-1, 1))
        rms_a = statistics.mean(r['atv_rms_ev_atom'] for r in rows)
        rms_o = statistics.mean(r['openmm_rms_ev_atom'] for r in rows)
        comparison[f'{dt:g}'] = dict(atv_mean=statistics.mean(a), openmm_mean=statistics.mean(o),
                                     pooled_se=se, atv_rms=rms_a, openmm_rms=rms_o,
                                     drift_agrees=abs(statistics.mean(a)-statistics.mean(o)) <= 2*se,
                                     rms_agrees=abs(rms_a/rms_o-1) <= 0.2)
    summary = dict(precision=args.precision, N=len(d['atoms']), temperature_K=args.temperature,
                   time_fs=args.drift_time, seeds=args.seeds, results=results, comparison=comparison,
                   accepted=all(c['drift_agrees'] and c['rms_agrees'] for c in comparison.values()))
    (out/'drift.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


def main():
    benchmark_start=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--ff',required=True)
    p.add_argument('--atv',required=True);p.add_argument('--out',required=True)
    p.add_argument('--replicate',type=int,default=1)
    p.add_argument('--skew',action='store_true',help='tilted supercell of the same replicated system (A=n a, B=a+n b, C=b+n c)')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--budget',type=float,default=10)
    p.add_argument('--measure-memory',action='store_true',help='sample process GPU memory and host peak RSS with an external monitor')
    p.add_argument('--prepared-json',help='internal OpenMM child input; avoids repeating validated topology replication')
    p.add_argument('--alpha',type=float,default=0.35);p.add_argument('--cutoff',type=float,default=9)
    p.add_argument('--spacing',type=float,default=1);p.add_argument('--dt',type=float,default=0.25)
    p.add_argument('--precision',choices=('double','mixed'),default='double')
    p.add_argument('--engine',choices=('compare','openmm'),default='compare')
    p.add_argument('--platform',choices=('CUDA','Reference'),default='CUDA')
    p.add_argument('--term',choices=('all','nonbonded','bond','angle','dihedral'),default='all')
    p.add_argument('--drift',action='store_true',help='NVE energy drift at dt and dt/2 against OpenMM (multiple seeds)')
    p.add_argument('--seeds',default='1,2,3');p.add_argument('--temperature',type=float,default=300)
    p.add_argument('--drift-time',type=float,default=2000,help='simulated time per run [fs]')
    p.add_argument('--drift-every',type=float,default=10,help='energy sampling interval [fs]')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.prepared_json and args.engine!='openmm':
        raise ValueError('--prepared-json is only valid for the OpenMM child')
    d=json.loads(Path(args.prepared_json).read_text()) if args.prepared_json else load(args.data,args.ff,args.replicate,args.skew)
    position_quantum=None
    if args.precision=='mixed' and not args.prepared_json:
        # Float-representable Angstrom values need not remain representable
        # after conversion to OpenMM's nm. Use a shared binary lattice:
        # x_A=5*k*2**e, x_nm=k*2**(e-1), with abs(5*k) < 2**24.
        # Both force kernels then start from exactly the same geometry.
        # Later trajectory precision remains independently checked.
        import numpy as np
        positions=np.asarray([a[:3] for a in d['atoms']],dtype=np.float64)
        cell=np.asarray(d['cell'],dtype=np.float64)
        maximum=max(float(np.max(np.abs(positions))),float(np.max(np.abs(cell))))
        position_quantum=5*2.0**(math.ceil(math.log2(maximum))-24)
        positions=np.rint(positions/position_quantum)*position_quantum
        cell=np.rint(cell/position_quantum)*position_quantum
        for values in (positions,cell):
            if not (np.array_equal(values,values.astype(np.float32).astype(np.float64))
                    and np.array_equal(values/10,(values/10).astype(np.float32).astype(np.float64))):
                raise ValueError('mixed input is not exactly representable in both Angstrom and nm')
        d['cell']=[float(v) for v in cell]
        d['box']=[d['cell'][0],d['cell'][4],d['cell'][8]]
        for atom,position in zip(d['atoms'],positions):
            atom[:3]=[float(v) for v in position]
    if args.drift:
        run_drift(d,args,out);return
    if args.engine=='openmm':
        print(json.dumps(openmm_run(d,args,out/'openmm.txt')));return
    prepare(d,out/'input.txt',args)
    # Both children consume the same already validated geometry/topology. This
    # cache only removes duplicate input replication, not OpenMM force work.
    (out/'system.json').write_text(json.dumps(d,separators=(',',':')))
    atv, atv_wall, atv_resources=run_engine([args.atv,str(out/'input.txt'),str(out/'atv.txt'),str(args.steps),str(args.warmup),args.term,str(args.samples)],args.budget,args.measure_memory)
    print(atv.stdout,end='');(out/'atv.log').write_text(atv.stdout+atv.stderr)
    sizes={name:int(re.search(name+r'=(\d+)',atv.stdout)[1]) for name in ('real_bytes','work_bytes','accum_bytes')}
    expected=4 if args.precision=='mixed' else 8
    if sizes!={'real_bytes':expected,'work_bytes':expected,'accum_bytes':8}:
        raise RuntimeError(f'Atomiverse binary precision mismatch: {sizes}')
    cmd=[sys.executable,*sys.argv,'--engine','openmm','--prepared-json',str(out/'system.json')]
    ref, ref_wall, ref_resources=run_engine(cmd,args.budget,args.measure_memory)
    print(ref.stdout,end='');(out/'openmm.log').write_text(ref.stdout+ref.stderr)
    import numpy as np
    a=np.loadtxt(out/'atv.txt',skiprows=1);b=np.loadtxt(out/'openmm.txt',skiprows=1)
    ea=float((out/'atv.txt').read_text().splitlines()[0]);eb=float((out/'openmm.txt').read_text().splitlines()[0])
    if not (np.isfinite(a).all() and np.isfinite(b).all() and math.isfinite(ea) and math.isfinite(eb)):
        raise RuntimeError('non-finite initial energy or force')
    err=float(np.max(np.abs(a-b)));rel=err/max(1,float(np.max(np.abs(b))))
    summary=dict(N=len(a),energy_atv_ev=ea,energy_openmm_ev=eb,energy_error_ev=ea-eb,
                 max_force_error_ev_A=err,relative_max_force_error=rel,
                 atv_process_seconds=atv_wall,openmm_process_seconds=ref_wall,
                 atv=atv.stdout.strip(),openmm=json.loads(ref.stdout))
    summary['protocol']=dict(steps=args.steps,warmup=args.warmup,samples=args.samples,dt_fs=args.dt,cutoff_A=args.cutoff,neighbor_skin_A=0.08*args.cutoff,alpha_inv_A=args.alpha,grid_spacing_A=args.spacing,cell_A=d['cell'],skew=args.skew,precision=args.precision,term=args.term,initial_position_storage='exact float32 in Angstrom and nm' if args.precision=='mixed' else 'float64',position_quantum_A=position_quantum)
    summary['protocol']['neighbor_reserve']='default build-time density/initial-count headroom'
    if args.measure_memory:
        if atv_resources['gpu_uuids'] != ref_resources['gpu_uuids']:
            raise RuntimeError('engines ran on different GPUs')
        summary['resources']={'atv':atv_resources,'openmm':ref_resources}
    summary['inputs_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name,path in (('data',args.data),('ff',args.ff),('prepared_input',out/'input.txt'),('openmm_input',out/'system.json'),('atv_binary',args.atv))}
    summary['atv_ms_per_step']=float(re.search(r'ms_per_step=(\S+)',atv.stdout)[1])
    summary['atv_final_energy_ev']=float(re.search(r' energy=(\S+)',atv.stdout)[1])
    if not math.isfinite(summary['atv_final_energy_ev']):
        raise RuntimeError('Atomiverse final energy is non-finite')
    pa=np.loadtxt(out/'atv.txt.final');pb=np.loadtxt(out/'openmm.txt.final')
    delta=pa-pb
    cell=np.asarray(d['cell']).reshape(3,3)
    fractional=delta@np.linalg.inv(cell)
    delta=(fractional-np.rint(fractional))@cell
    summary['max_final_position_error_A']=float(np.max(np.abs(delta)))
    summary['atv_over_openmm']=summary['atv_ms_per_step']/summary['openmm']['ms_per_step']
    summary['final_energy_error_ev']=summary['atv_final_energy_ev']-summary['openmm']['final_energy_ev']
    summary['validation']={
        'initial_energy':abs(ea-eb)<=max(1e-3,abs(eb)*3e-5),
        'initial_forces':rel<=(1e-4 if args.precision=='mixed' else 3e-5),
        'final_positions':summary['max_final_position_error_A']<=(1e-2 if args.precision=='mixed' else 1e-4),
        'final_energy':abs(summary['final_energy_error_ev'])<=max(1e-3,abs(summary['openmm']['final_energy_ev'])*3e-5),
    }
    summary['accepted']=all(summary['validation'].values())
    summary['benchmark_seconds']=time.perf_counter()-benchmark_start
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    if not summary['accepted']:
        raise RuntimeError(f"OpenMM parity failed: {summary['validation']}")



if __name__=='__main__':
    main()
