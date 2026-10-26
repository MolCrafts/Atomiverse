"""HF declarations and public engine adapters; each timed call owns fresh state.

No integral, molecule, density, solver or force oracle is prepared by an adapter
factory. Factories only import/configure runtimes and fingerprint their binaries.
The GHF Neel matrix is the declared input seed, assembled inside the timed call.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
from pathlib import Path

from benchmarks.framework.provenance import file_hash
from benchmarks.framework.specification import Check, Participant, Row, System

HARTREE_IN_EV = 27.211386245988
BOHR_IN_ANGSTROM = 1.0 / 1.8897261245650618


class BindingMissing(RuntimeError):
    benchmark_status = "binding_missing"


class CapabilityUnavailable(RuntimeError):
    benchmark_status = "not_implemented"


class NotConverged(RuntimeError):
    benchmark_status = "not_converged"


class ScientificFailure(RuntimeError):
    benchmark_status = "scientific_failure"


def systems(*names: str) -> list[System]:
    canonical = json.loads((Path(__file__).resolve().parents[1] / "systems/hf.json").read_text())
    return [System.model_validate(canonical[name]) for name in names]


def participants(method: str, basis: str, task: str, guess: str) -> list[Participant]:
    common = dict(entrypoint="benchmarks.scf.hf:atv_adapter", engine="atomiverse", role="atv")
    direct_reason = ("Native GHF requires stored dense ERI for nonsymmetric spin-off-diagonal exchange; direct spinor SCF is not implemented"
                     if method == "GHF" else None)
    result = [Participant(id="atv.cpu.direct", backend="cpu", required=method != "GHF",
                          available=direct_reason is None, unavailable_reason=direct_reason,
                          options={"integral_mode": "direct"}, **common)]
    if method in {"RHF", "GHF"}:
        result.append(Participant(id="atv.cpu.stored", backend="cpu", required=method == "GHF",
                                  options={"integral_mode": "stored"}, **common))
    cuda_reason = None
    if method not in {"RHF", "UHF"}:
        cuda_reason = f"CUDA {method} is not implemented"
    elif method == "UHF" and (basis != "STO-3G" or guess != "minao"):
        cuda_reason = "CUDA UHF public envelope is STO-3G with its MINAO start"
    elif task == "forces" and basis != "STO-3G":
        cuda_reason = "CUDA analytic HF forces outside STO-3G s/p are not implemented"
    result.append(Participant(id="atv.cuda", backend="cuda", available=cuda_reason is None,
                              unavailable_reason=cuda_reason, **common))
    result.extend([
        Participant(id="pyscf.cpu", engine="pyscf", backend="cpu", role="oracle", required=True,
                    entrypoint="benchmarks.scf.hf:pyscf_adapter"),
        Participant(id="gpu4pyscf.cuda", engine="gpu4pyscf", backend="cuda", role="oracle",
                    entrypoint="benchmarks.scf.hf:pyscf_adapter"),
    ])
    return result


def row(method: str, suffix: str, task: str, basis: str, names: list[str], *,
        guess: str = "hcore", energy_atol: float = 1e-8, neel: bool = False) -> Row:
    model = dict(method=method, comparison_class="matched_accuracy", basis=basis, cartesian=basis in {"STO-3G", "6-31G", "6-31G*"},
                 init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-8, max_cycle=200,
                 damping=0.7 if method == "GHF" else 0.0,
                 diis_space=6 if method == "GHF" else 8,
                 diis_start_cycle=5 if method == "GHF" else 1,
                 nonzero_force=task == "forces", neel=neel,
                 spin_rotation=2 * math.pi / 3 if neel else 0.0)
    checks = [Check(observable="energy_hartree", unit="Ha", atol=energy_atol,
                    justification="Live independent SCF reference; no archived golden replaces this call.")]
    if task == "forces":
        checks.append(Check(observable="forces_hartree_per_bohr", unit="Ha/Bohr", atol=1e-6,
                            justification="Analytic force components; sign and unit conversion are explicit."))
    if neel:
        checks.append(Check(observable="spin_mixing_norm", unit="dimensionless", atol=1e-4,
                            justification="Converged alpha/beta AO density block; both engines must also exceed 0.1."))
    selected_systems = systems(*names)
    if basis != "STO-3G":
        for selected in selected_systems:
            selected.tiers = [tier for tier in selected.tiers if tier != "smoke"]
    capability = f"hf.{method.lower()}.{'gradient' if task == 'forces' else 'energy'}"
    return Row(id=f"{method.lower()}.{suffix}", label=f"{method} {basis} {task} ({guess})",
               capability=capability, task=task, model=model, systems=selected_systems,
               participants=participants(method, basis, task, guess), reference="pyscf.cpu",
               checks=checks, cuda_only=True)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version({"molpy": "molcrafts-molpy"}.get(name, name))
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(engine, module, threadpoolctl, threads: int, **config) -> dict:
    paths = set()
    module_path = getattr(module, "__file__", None)
    if module_path:
        paths.add(str(Path(module_path).resolve()))
    pools = threadpoolctl.threadpool_info()
    paths.update(p["filepath"] for p in pools if p.get("filepath"))
    identities = {p: file_hash(p) for p in sorted(paths) if Path(p).is_file()}
    return dict(engine=engine, package_version=_version(engine), molpy_version=_version("molpy"),
                binary_path=module_path, binary_sha256=identities.get(str(Path(module_path).resolve())) if module_path else None,
                binaries_sha256=identities, requested_threads=threads, threadpools=pools,
                versions={name: _version(name) for name in [engine, "numpy", "scipy", "molpy", "threadpoolctl"]},
                actual_engine_config=config)


def _thread_mismatch(pools: list[dict], threads: int) -> bool:
    return any(p.get("num_threads") is not None and p["num_threads"] != threads for p in pools)


def _shared_libraries(root: Path, *, recursive: bool = False) -> list[Path]:
    discover = root.rglob if recursive else root.glob
    return sorted({file for pattern in ("*.so", "*.so.[0-9]*", "*.dylib")
                   for file in discover(pattern) if file.is_file()})


def atv_adapter(row: Row, system: System, participant: Participant, threads: int):
    if not participant.available:
        raise CapabilityUnavailable(participant.unavailable_reason or "Declared native capability unavailable")
    np = importlib.import_module("numpy")
    molpy = importlib.import_module("molpy")
    scf = importlib.import_module("atomiverse.scf")
    threadpoolctl = importlib.import_module("threadpoolctl")
    name = ("Cuda" if participant.backend == "cuda" else "") + row.model["method"]
    cls = getattr(scf, name, None)
    if cls is None:
        raise BindingMissing(f"Public atomiverse.scf.{name} is absent from the loaded binary")
    module = importlib.import_module(cls.__module__)
    limits = threadpoolctl.threadpool_limits(limits=threads)
    backend_info = scf.cuda_backend_info() if participant.backend == "cuda" else scf.backend_info()
    meta = _metadata("atomiverse", module, threadpoolctl, threads, backend=participant.backend,
                     backend_info=backend_info, integral_mode=participant.options.get("integral_mode", "device_dense"))
    meta["requested_model"] = row.model
    meta["comparison_class"] = "matched_accuracy"
    meta["resource_mismatch"] = _thread_mismatch(meta["threadpools"], threads) or (
        participant.backend == "cpu" and backend_info.get("openmp_threads") != threads)
    if participant.backend == "cuda" and row.model["method"] == "GHF":
        raise BindingMissing("CUDA GHF is not a public implemented engine")

    def run():
        _ = limits  # Keep the configured runtime controller alive for every call.
        frame = molpy.Frame()
        atoms = molpy.Block()
        xyz = np.asarray(system.positions_angstrom, dtype=np.float64)
        for index, key in enumerate(["x", "y", "z"]):
            atoms.insert(key, xyz[:, index].copy())
        atoms.insert("atomic_number", np.asarray(system.atomic_numbers, dtype=np.uint32))
        # Mass is input completeness only; it does not enter molecular HF.
        atoms.insert("mass", np.asarray([1.008 if z == 1 else 15.999 for z in system.atomic_numbers], dtype=np.float64))
        frame["atoms"] = atoms
        frame.box = molpy.Box.cube(max(40.0, float(np.ptp(xyz, axis=0).max()) + 24.0))
        ne = sum(system.atomic_numbers) - system.charge
        kwargs = dict(basis=row.model["basis"], tol=row.model["conv_tol"], max_iter=row.model["max_cycle"])
        method = row.model["method"]
        if method == "GHF":
            kwargs.update(n_electrons=ne, spin_rotation=row.model["spin_rotation"])
        else:
            kwargs["charge"] = system.charge
            if method in {"UHF", "ROHF"}:
                kwargs.update(n_alpha=(ne + system.spin) // 2, n_beta=(ne - system.spin) // 2)
        solver = cls(**kwargs)
        try:
            if participant.backend == "cpu":
                solver.integral_mode = participant.options["integral_mode"]
                solver.init_guess = row.model["init_guess"]
                for field in ["damping", "conv_tol_grad", "diis_space", "diis_start_cycle"]:
                    setattr(solver, field, row.model[field])
            if row.task == "forces":
                force_ev_a = np.asarray(solver.forces(frame), dtype=np.float64).copy()
                from_frame = np.column_stack([np.asarray(frame["atoms"][k]) for k in ["fx", "fy", "fz"]])
                if not np.allclose(force_ev_a, from_frame, rtol=0, atol=1e-8):
                    raise ScientificFailure("Frame force columns disagree with the public forces() result")
            else:
                solver.run(frame)
            if participant.backend == "cuda":
                if not solver.last_scf_on_device():
                    raise ScientificFailure("CUDA HF reported a CPU SCF fallback")
                if row.task == "forces" and not solver.last_force_on_device():
                    raise ScientificFailure("CUDA force workload reported a CPU force fallback")
            energy = float(solver.energy())
            if not solver.converged() or not np.isfinite(energy):
                raise NotConverged(f"{name} did not return a finite converged SCF")
            frame_energy = float(frame.meta["energy"])
            if not np.isfinite(frame_energy) or abs(frame_energy / HARTREE_IN_EV - energy) > 1e-10:
                raise ScientificFailure("Frame energy does not match the solver Hartree total")
            observables = {"energy_hartree": energy}
            if row.task == "forces":
                forces = force_ev_a * (BOHR_IN_ANGSTROM / HARTREE_IN_EV)
                if forces.shape != (len(system.atomic_numbers), 3) or not np.isfinite(forces).all():
                    raise ScientificFailure("Invalid analytic force array")
                if row.model["nonzero_force"] and float(np.linalg.norm(forces)) < 1e-8:
                    raise ScientificFailure("Nonzero-force workload returned an all-zero force")
                observables["forces_hartree_per_bohr"] = forces.tolist()
            if row.model["neel"]:
                if not hasattr(solver, "spin_density_block"):
                    raise BindingMissing("GHF spin_density_block is required to certify noncollinearity")
                norm = float(np.linalg.norm(np.asarray(solver.spin_density_block("alpha", "beta"))))
                if not np.isfinite(norm) or norm <= 0.1:
                    raise ScientificFailure("H3 GHF remained in the collinear saddle")
                observables["spin_mixing_norm"] = norm
            observed = dict(iterations=int(solver.iterations()), converged=True,
                            initial_guess=row.model["init_guess"], spin_rotation=row.model["spin_rotation"],
                            solver_type=type(solver).__name__, energy_only=row.task != "forces",
                            tol=float(solver.tol), max_iter=int(solver.max_iter),
                            basis=solver.basis, charge=solver.charge, final_residuals="not_exposed")
            if participant.backend == "cpu":
                observed.update({key: getattr(solver, key) for key in ["integral_mode", "init_guess", "damping", "conv_tol_grad", "diis_space", "diis_start_cycle"]})
                observed["density_criterion"] = "not_exposed"
            else:
                tol = float(solver.tol)
                observed.update(
                    initial_guess="hcore" if method == "RHF" else "minao",
                    initial_guess_observability="documented native policy for declared H/O systems; no public guess getter",
                    level_shift=float(solver.level_shift),
                    density_residual_tol=tol, abs_delta_energy_hartree_tol=tol, orbital_residual_tol=math.sqrt(tol),
                    criterion_observability="native CUDA convergence contract; final residual values not exposed",
                    conv_tol_grad="not_exposed; requested CPU control not applied",
                    damping="not_exposed", diis_space="not_exposed", diis_start_cycle="not_exposed",
                    last_scf_on_device=bool(solver.last_scf_on_device()),
                    last_force_on_device=bool(solver.last_force_on_device()) if row.task == "forces" else "not_requested")
            return {"observables": observables, "metadata": {**meta, "resolved_settings": observed}}
        finally:
            del solver, frame
            if participant.backend == "cuda" and hasattr(scf, "cuda_synchronize"):
                scf.cuda_synchronize()
    return run


def pyscf_adapter(row: Row, system: System, participant: Participant, threads: int):
    np = importlib.import_module("numpy")
    pyscf = importlib.import_module("pyscf")
    gto = importlib.import_module("pyscf.gto")
    threadpoolctl = importlib.import_module("threadpoolctl")
    gpu = participant.backend == "cuda"
    engine_scf = importlib.import_module("gpu4pyscf.scf" if gpu else "pyscf.scf")
    cls = getattr(engine_scf, row.model["method"], None)
    if cls is None:
        raise BindingMissing(f"{participant.engine} public {row.model['method']} is not implemented")
    cp = importlib.import_module("cupy") if gpu else None
    if cp is not None:
        cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device().synchronize()
    limits = threadpoolctl.threadpool_limits(limits=threads)
    pyscf.lib.num_threads(threads)
    meta = _metadata(participant.engine, engine_scf, threadpoolctl, threads,
                     backend=participant.backend, pyscf_version=pyscf.__version__,
                     pyscf_num_threads=pyscf.lib.num_threads())
    meta["requested_model"] = row.model
    meta["comparison_class"] = "matched_accuracy"
    meta["entrypoint_path"] = getattr(engine_scf, "__file__", None)
    # Include the integral/JK shared objects, not just the Python module path.
    for file in _shared_libraries(Path(pyscf.__file__).resolve().parent / "lib"):
        meta["binaries_sha256"][str(file)] = file_hash(file)
    if gpu:
        gpu_package = importlib.import_module("gpu4pyscf")
        for file in _shared_libraries(Path(gpu_package.__file__).resolve().parent, recursive=True):
            meta["binaries_sha256"][str(file)] = file_hash(file)
    primary = next((path for path in meta["binaries_sha256"] if ("libgvhf" if gpu else "libcgto") in path), None)
    meta["binary_path"] = primary
    meta["binary_sha256"] = meta["binaries_sha256"].get(primary)
    meta["resource_mismatch"] = _thread_mismatch(meta["threadpools"], threads) or pyscf.lib.num_threads() != threads
    if cp is not None:
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        meta["actual_engine_config"]["cuda"] = dict(
            device=int(cp.cuda.runtime.getDevice()), name=device_name,
            runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            driver_version=int(cp.cuda.runtime.driverGetVersion()),
            total_memory_bytes=int(properties["totalGlobalMem"]))

    def run():
        _ = limits
        mol = gto.M(atom=list(zip(system.atomic_numbers, system.positions_angstrom)), basis=row.model["basis"],
                    charge=system.charge, spin=system.spin, unit="Angstrom", cart=row.model["cartesian"], verbose=0)
        mf = cls(mol)
        if gpu and not type(mf).__module__.startswith("gpu4pyscf"):
            raise ScientificFailure("GPU oracle returned a non-GPU solver")
        mf.conv_tol = row.model["conv_tol"]
        mf.conv_tol_grad = row.model["conv_tol_grad"]
        mf.max_cycle = row.model["max_cycle"]
        mf.diis_space = row.model["diis_space"]
        mf.diis_start_cycle = row.model["diis_start_cycle"]
        mf.damp = row.model["damping"]
        mf.init_guess = "1e" if row.model["init_guess"] == "hcore" else "minao"
        try:
            dm0 = None
            if row.model["neel"]:
                # Canonical localized Neel input, not an alternative GHF solver:
                # one up moment per H AO, rotated by theta_mu=2*pi*mu/3.
                n = mol.nao_nr()
                if n != 3 or sum(system.atomic_numbers) - system.charge != 3:
                    raise ScientificFailure("Localized H3 seed requires three STO-3G H AOs")
                dm0 = np.zeros((2*n, 2*n))
                for mu in range(n):
                    c = math.cos(row.model["spin_rotation"] * mu / 2)
                    s = math.sin(row.model["spin_rotation"] * mu / 2)
                    dm0[mu, mu], dm0[n+mu, n+mu] = c*c, s*s
                    dm0[mu, n+mu] = dm0[n+mu, mu] = c*s
                if cp is not None:
                    dm0 = cp.asarray(dm0)
            energy = float(mf.kernel(dm0=dm0))
            if not mf.converged or not np.isfinite(energy):
                raise NotConverged(f"{participant.engine} {row.model['method']} failed to converge")
            observables = {"energy_hartree": energy}
            if row.task == "forces":
                gradient = mf.nuc_grad_method()
                if gpu and not type(gradient).__module__.startswith("gpu4pyscf"):
                    raise ScientificFailure("GPU oracle gradient falls back to CPU")
                forces = -gradient.kernel()
                del gradient
                if cp is not None:
                    forces = cp.asnumpy(forces)
                forces = np.asarray(forces, dtype=float)
                if not np.isfinite(forces).all() or forces.shape != (len(system.atomic_numbers), 3):
                    raise ScientificFailure("Invalid independent analytic force")
                if row.model["nonzero_force"] and float(np.linalg.norm(forces)) < 1e-8:
                    raise ScientificFailure("Independent nonzero-force workload returned all zeros")
                observables["forces_hartree_per_bohr"] = forces.tolist()
            if row.model["neel"]:
                dm = mf.make_rdm1()
                if cp is not None:
                    dm = cp.asnumpy(dm)
                norm = float(np.linalg.norm(dm[:mol.nao_nr(), mol.nao_nr():]))
                if norm <= 0.1 or not np.isfinite(norm):
                    raise ScientificFailure("Independent H3 GHF oracle remained collinear")
                observables["spin_mixing_norm"] = norm
            settings = dict(initial_guess=mf.init_guess, damping=float(mf.damp), diis_space=int(mf.diis_space),
                            diis_start_cycle=int(mf.diis_start_cycle), conv_tol=float(mf.conv_tol),
                            conv_tol_grad=float(mf.conv_tol_grad), cartesian=bool(mol.cart), nao=int(mol.nao_nr()),
                            converged=True, iterations=getattr(mf, "cycles", None),
                            direct_scf=bool(mf.direct_scf), max_memory_MiB=float(mf.max_memory),
                            level_shift=float(mf.level_shift), final_residuals="not_exposed_by_adapter",
                            solver_type=type(mf).__module__ + "." + type(mf).__name__)
            if cp is not None:
                cp.cuda.Device().synchronize()
            return {"observables": observables, "metadata": {**meta, "resolved_settings": settings}}
        finally:
            del mf, mol
            if cp is not None:
                cp.cuda.Device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()
    return run
