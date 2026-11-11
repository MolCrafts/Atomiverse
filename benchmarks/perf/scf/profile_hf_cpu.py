"""Diagnostic CPU HF comparison, including exact-J/K phase logs.

Run on a compute allocation after building bench_hf_cpu (benchmarks/perf):
  profile_hf_cpu.py --binary $BUILD/bin/bench_hf_cpu --out DIR [--validate]
This is separate from the Python benchmark framework; native C++ and PySCF
entry boundaries are recorded explicitly. DIR must not exist; keep it outside
the source tree or under build/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


def atoms(system, waters):
    if system == "water":
        return [(z, (4.0 * w, y, zpos)) for w in range(waters)
                for z, y, zpos in [(8, 0, .1173), (1, .7572, -.4692), (1, -.7572, -.4692)]]
    if system == "h3":
        return [(1, (0, 0, 0)), (1, (1, 0, 0)), (1, (.5, 3**.5 / 2, 0))]
    return [(8 if system == "o2" else 1, (i * (1.207 if system == "o2" else .74), 0, 0))
            for i in range(2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("positive threads and repeats required")
    allocation_cpus = (os.sched_getaffinity(0)
                       if hasattr(os, "sched_getaffinity") else None)
    if allocation_cpus is not None and args.threads > len(allocation_cpus):
        parser.error("requested threads exceed the allocated CPU set")
    args.out.mkdir(parents=True, exist_ok=False)
    # Set resource controls before importing scientific libraries.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import pyscf
    from pyscf import gto, scf, lib
    from threadpoolctl import threadpool_info, threadpool_limits
    lib.num_threads(args.threads)
    limits = threadpool_limits(limits=args.threads)
    cases = [("RHF", "sto3g", "water", 16, False, memory) for memory in (4000.0, 0.0)]
    if args.validate:
        cases = [(method, basis, system, 1, True, memory)
                 for method, system in [("RHF", "water"), ("UHF", "o2"), ("ROHF", "o2")]
                 for basis in ("sto3g", "def2svp") for memory in (4000.0, 0.0)]
        cases += [("GHF", "sto3g", system, 1, False, memory)
                  for system in ("h2", "h3") for memory in (4000.0, 0.0)]
    cases += [("RHF", "def2svp", "water", count, args.validate, memory)
              for count in (4, 8) for memory in (4000.0, 0.0)]
    records = []
    for method, basis, system, waters, forces, memory in cases:
        label = f"{method}-{basis}-{system}{waters}-cache{memory:g}"
        command = [str(args.binary.resolve()), method, basis, system, str(waters),
                   str(args.threads), str(args.repeats), str(memory), str(int(forces))]
        env = dict(os.environ, ATV_LOG_LEVEL="debug")
        # OpenMP can pin the Python caller to one core. Let the native
        # subprocess inherit the full allocation before it binds its team.
        parent_cpus = None
        if allocation_cpus is not None:
            parent_cpus = os.sched_getaffinity(0)
            os.sched_setaffinity(0, allocation_cpus)
        try:
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=1200)
        finally:
            if parent_cpus is not None:
                os.sched_setaffinity(0, parent_cpus)
        (args.out / f"{label}.log").write_text(result.stdout)
        result.check_returncode()
        native, build_ms, caches = [], [], []
        for line in result.stdout.splitlines():
            build = re.search(r"\[exact_jk\] cache_bytes=(\d+) build_ms=([\d.]+)", line)
            if build:
                caches.append(int(build[1]))
                build_ms.append(float(build[2]))
            elif line.startswith('{"repeat"'):
                if not caches or bool(caches[0]) != (memory > 0):
                    raise RuntimeError(f"{label}: ATV used an unexpected SCF integral strategy")
                sample = json.loads(line)
                if sample["threads"] != args.threads:
                    raise RuntimeError(f"{label}: ATV used an unexpected thread count")
                sample["eri_cache_bytes"] = caches[0]
                # ROHF forces can prepare a separate exchange operator.
                sample["auxiliary_eri_cache_bytes"] = caches[1:]
                native.append(sample)
                caches.clear()
        if len(native) != args.repeats or caches:
            raise RuntimeError("missing or incomplete native samples")
        oracle = []
        for repeat in range(args.repeats):
            start = time.perf_counter()
            mol = gto.M(atom=atoms(system, waters), basis="sto-3g" if basis == "sto3g" else "def2-svp",
                        spin=2 if system == "o2" else 1 if system == "h3" else 0,
                        cart=basis == "sto3g", max_memory=memory, verbose=0)
            mol.incore_anyway = False
            mf = getattr(scf, method)(mol)
            mf.direct_scf, mf.direct_scf_tol = True, 1e-13
            mf.chkfile = None
            mf.init_guess = "1e"
            mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-8, 200
            mf.diis_space = 6 if method == "GHF" else 8
            mf.diis_start_cycle = 5 if method == "GHF" else 1
            mf.damp = .7 if method == "GHF" else 0
            seed = None
            if method == "GHF" and system == "h3":
                seed = np.zeros((6, 6))
                for mu in range(3):
                    c, s = np.cos(np.pi * mu / 3), np.sin(np.pi * mu / 3)
                    seed[mu, mu], seed[3+mu, 3+mu] = c*c, s*s
                    seed[mu, 3+mu] = seed[3+mu, mu] = c*s
            energy = float(mf.kernel(dm0=seed))
            if not mf.converged:
                raise RuntimeError("PySCF did not converge")
            sample = dict(repeat=repeat, energy_hartree=energy, iterations=mf.cycles,
                          eri_cache_bytes=0 if mf._eri is None else mf._eri.nbytes,
                          direct_scf_initialized=mf._opt.get(None) is not None)
            if forces:
                sample["forces_hartree_per_bohr"] = (-mf.nuc_grad_method().kernel()).tolist()
            del mf, mol
            sample["seconds"] = time.perf_counter() - start
            if (bool(sample["eri_cache_bytes"]) != (memory > 0)
                    or (memory == 0 and not sample["direct_scf_initialized"])):
                raise RuntimeError(f"{label}: PySCF used an unexpected integral strategy")
            oracle.append(sample)
        energy_error = max(abs(a["energy_hartree"] - b["energy_hartree"])
                           for a, b in zip(native, oracle))
        force_error = (max(float(np.max(np.abs(np.array(a["forces_hartree_per_bohr"])
                                             - b["forces_hartree_per_bohr"])))
                           for a, b in zip(native, oracle)) if forces else None)
        passed = energy_error <= (1e-6 if waters == 16 else 1e-8) and (force_error is None or force_error <= 1e-6)
        record = dict(case=label, command=command, memory_mb=memory, native=native, pyscf=oracle,
                      max_energy_error_ha=energy_error, max_force_error_ha_bohr=force_error,
                      passed=passed,
                      build_ms=build_ms,
                      jk_ms=[float(x) for x in re.findall(r"\[exact_jk\] jk_ms=([\d.]+)", result.stdout)],
                      loop_ms=[float(x) for x in re.findall(r"\[scf\] iterations=\d+ loop_ms=([\d.]+)", result.stdout)])
        records.append(record)
        print(label, "PASS" if passed else "FAIL", energy_error, force_error, flush=True)
        evidence = dict(hardware=platform.uname()._asdict(),
                        allocation_cpus=sorted(allocation_cpus) if allocation_cpus is not None else None,
                        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        scheduler={key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK")},
                        requested_threads=args.threads, native_binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        pyscf_version=pyscf.__version__, threadpools=threadpool_info(),
                        pyscf_checkpoint=False,
                        environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES")},
                        native_linkage=subprocess.run(["ldd", str(args.binary.resolve())], text=True, capture_output=True).stdout if platform.system() == "Linux" else None,
                        timing="fresh C++ Frame/method/run or forces/cleanup vs fresh PySCF molecule/solver/kernel/gradient/cleanup; imports excluded; first sample cold",
                        records=records)
        (args.out / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    _ = limits
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
