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


def load(data, ff, rep):
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
            raise ValueError("benchmark currently requires an orthorhombic cell")
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
    base_positions = [[(float(t[4+d])-bounds[d][0]) % lengths[d] for d in range(3)] for t in base]
    atoms, bonds, angles, dihedrals = [], [], [], []
    for ix in range(rep):
        for iy in range(rep):
            for iz in range(rep):
                cell = (ix, iy, iz)
                for t, pos in zip(base, base_positions):
                    typ = int(t[2])
                    atoms.append([pos[d]+cell[d]*lengths[d] for d in range(3)]+[float(t[3]), typ-1, mass[typ]])
                for name, out in (("Bonds", bonds), ("Angles", angles), ("Dihedrals", dihedrals)):
                    for t in sections[name]:
                        # Follow each bonded path in minimum-image geometry.
                        # A boundary-crossing endpoint belongs to the adjacent
                        # replica, not the same replica with a stretched bond.
                        ids = [int(a)-1 for a in t[2:]]
                        image = [0, 0, 0]
                        mapped = []
                        previous = ids[0]
                        for atom in ids:
                            for d in range(3):
                                image[d] -= round((base_positions[atom][d]-base_positions[previous][d])/lengths[d])
                            target = [(cell[d]+image[d]) % rep for d in range(3)]
                            mapped.append(((target[0]*rep+target[1])*rep+target[2])*len(base)+atom)
                            previous = atom
                        out.append(mapped+[int(t[1])-1])
    # Input geometry invariant: replication must preserve every consecutive
    # bonded displacement, not merely the atom count or total charge.
    for name, records in (("Bonds", bonds), ("Angles", angles), ("Dihedrals", dihedrals)):
        source = sections[name]
        for index, record in enumerate(records):
            ids = [int(a)-1 for a in source[index % len(source)][2:]]
            for edge in range(len(ids)-1):
                for d in range(3):
                    ref_delta = base_positions[ids[edge+1]][d]-base_positions[ids[edge]][d]
                    ref_delta -= lengths[d]*round(ref_delta/lengths[d])
                    new_delta = atoms[record[edge+1]][d]-atoms[record[edge]][d]
                    new_delta -= lengths[d]*rep*round(new_delta/(lengths[d]*rep))
                    if abs(new_delta-ref_delta) > 1e-9:
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
                box=[l*rep for l in lengths], **{k:[v[t] for t in sorted(v)] for k,v in coeff.items()})


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
    rows = [[len(d['atoms']), *d['box'], len(d['pair']), len(d['bond']), len(d['angle']),
             len(d['dihedral']), len(d['bonds']), len(d['angles']), len(d['dihedrals']),
             args.alpha, args.cutoff, args.spacing, 5, args.dt]]
    for name in ('pair','bond','angle'):
        rows.extend(d[name])
    rows.extend(polynomial(t) for t in d['dihedral'])
    for name in ('atoms','bonds','angles','dihedrals'):
        rows.extend(d[name])
    path.write_text('\n'.join(' '.join(map(str,r)) for r in rows)+'\n')


def openmm_run(d, args, result):
    import openmm as mm
    import numpy as np
    from openmm import unit
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*[l/10 if i==j else 0 for j in range(3)])
                                         for i,l in enumerate(d['box'])])
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


def main():
    benchmark_start=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--ff',required=True)
    p.add_argument('--atv',required=True);p.add_argument('--out',required=True)
    p.add_argument('--replicate',type=int,default=1)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--budget',type=float,default=10)
    p.add_argument('--measure-memory',action='store_true',help='sample memory in separate engine processes after unmonitored timing')
    p.add_argument('--prepared-json',help='internal OpenMM child input; avoids repeating validated topology replication')
    p.add_argument('--alpha',type=float,default=0.35);p.add_argument('--cutoff',type=float,default=9)
    p.add_argument('--spacing',type=float,default=1);p.add_argument('--dt',type=float,default=0.25)
    p.add_argument('--precision',choices=('double','mixed'),default='double')
    p.add_argument('--engine',choices=('compare','openmm'),default='compare')
    p.add_argument('--platform',choices=('CUDA','Reference'),default='CUDA')
    p.add_argument('--term',choices=('all','nonbonded','bond','angle','dihedral'),default='all')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.prepared_json and args.engine!='openmm':
        raise ValueError('--prepared-json is only valid for the OpenMM child')
    d=json.loads(Path(args.prepared_json).read_text()) if args.prepared_json else load(args.data,args.ff,args.replicate)
    position_quantum=None
    if args.precision=='mixed' and not args.prepared_json:
        # Float-representable Angstrom values need not remain representable
        # after conversion to OpenMM's nm. Use a shared binary lattice:
        # x_A=5*k*2**e, x_nm=k*2**(e-1), with abs(5*k) < 2**24.
        # Both force kernels then start from exactly the same geometry.
        # Later trajectory precision remains independently checked.
        import numpy as np
        positions=np.asarray([a[:3] for a in d['atoms']],dtype=np.float64)
        box=np.asarray(d['box'],dtype=np.float64)
        maximum=max(float(np.max(np.abs(positions))),float(np.max(np.abs(box))))
        position_quantum=5*2.0**(math.ceil(math.log2(maximum))-24)
        positions=np.rint(positions/position_quantum)*position_quantum
        box=np.rint(box/position_quantum)*position_quantum
        for values in (positions,box):
            if not (np.array_equal(values,values.astype(np.float32).astype(np.float64))
                    and np.array_equal(values/10,(values/10).astype(np.float32).astype(np.float64))):
                raise ValueError('mixed input is not exactly representable in both Angstrom and nm')
        d['box']=[float(v) for v in box]
        for atom,position in zip(d['atoms'],positions):
            atom[:3]=[float(v) for v in position]
    if args.engine=='openmm':
        print(json.dumps(openmm_run(d,args,out/'openmm.txt')));return
    prepare(d,out/'input.txt',args)
    # Both children consume the same already validated geometry/topology. This
    # cache only removes duplicate input replication, not OpenMM force work.
    (out/'system.json').write_text(json.dumps(d,separators=(',',':')))
    atv_command=[args.atv,str(out/'input.txt'),str(out/'atv.txt'),str(args.steps),str(args.warmup),args.term,str(args.samples)]
    atv_probe=atv_command.copy()
    atv_probe[2]=str(out/'memory-atv.txt')
    atv, atv_wall, atv_resources=run_engine(atv_command,args.budget,args.measure_memory,atv_probe)
    print(atv.stdout,end='');(out/'atv.log').write_text(atv.stdout+atv.stderr)
    sizes={name:int(re.search(name+r'=(\d+)',atv.stdout)[1]) for name in ('real_bytes','work_bytes','accum_bytes','motion_bytes','position_residual')}
    expected=4 if args.precision=='mixed' else 8
    if sizes!={'real_bytes':expected,'work_bytes':expected,'accum_bytes':8,'motion_bytes':8,'position_residual':int(args.precision=='mixed')}:
        raise RuntimeError(f'Atomiverse binary precision mismatch: {sizes}')
    cmd=[sys.executable,*sys.argv,'--engine','openmm','--prepared-json',str(out/'system.json')]
    ref_probe=[*cmd,'--out',str(out/'memory-openmm')]
    ref, ref_wall, ref_resources=run_engine(cmd,args.budget,args.measure_memory,ref_probe)
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
    summary['protocol']=dict(steps=args.steps,warmup=args.warmup,samples=args.samples,dt_fs=args.dt,cutoff_A=args.cutoff,neighbor_skin_A=0.08*args.cutoff,alpha_inv_A=args.alpha,grid_spacing_A=args.spacing,precision=args.precision,term=args.term,initial_position_storage='exact float32 in Angstrom and nm' if args.precision=='mixed' else 'float64',position_quantum_A=position_quantum)
    summary['protocol']['neighbor_reserve']='default build-time density/initial-count headroom'
    summary['protocol']['motion_bytes']=sizes['motion_bytes']
    summary['protocol']['position_representation']='float32+residual' if sizes['position_residual'] else 'float64'
    summary['protocol']['throughput_monitor']='none'
    summary['protocol']['resource_measurement']='separate_process' if args.measure_memory else 'none'
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
    box=np.asarray(d['box'])
    delta-=box*np.rint(delta/box)
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
