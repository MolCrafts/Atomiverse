/// @file cuda/planewave/pw_solver.cu
/// @brief cuda::pw::PW — GPU plane-wave SCF method object.
///
/// Mirrors the CPU SCF loop in src/cpu/planewave/pw_solver.cpp stage for
/// stage. Problem assembly (`setup_pw_problem`) and the per-k table binding
/// stay on the host — they are geometry work, not per-iteration work. Both
/// halves of the iteration are device-resident:
///
///   * band side — `internal::DeviceBandSolver` (cuFFT + cuBLAS Davidson),
///   * density side — `internal::DeviceDensity` (ρ(G) → V_H → V_xc (LSDA or
///     PBE GGA) → V_eff(r) → Pulay mixing → energy/residual reductions).
///
/// The two meet only through two device pointers bound once per run
/// (`bind_grids`), so **no O(N_G) or O(N_r) density-side array crosses PCIe
/// per SCF iteration**. The claim is scoped to `DeviceDensity` on purpose —
/// the band side still transfers per iteration (see below). What the density
/// side still crosses with: the Pulay Gram matrix (≤ 8²·16 B, only once the
/// history holds ≥ 2 entries), its ≤ 8 coefficients, and 5 energy/residual
/// doubles. Each is documented at its call site in device_density.cu.
///
/// The band side is geometry-hoisted too: `build_kpoints` uploads every per-k
/// table (G→FFT map, T(G), q = k+G, KB descriptors, initial Davidson subspace)
/// once per run, so `bind_kpoint` allocates nothing, copies nothing across
/// PCIe and synchronizes nothing.
///
/// The Davidson subspace eigenproblem runs on device too (cuSOLVER `Zheevd`,
/// in place), so the n_sub² Hamiltonian and the Ritz rotation no longer round
/// trip. Still NOT device-resident (named, not claimed fixed): the
/// host-pointer `cublasDznrm2` residual norms, the n_bands eigenvalues that
/// are kernel scalar arguments, and the Zheevd `devInfo` check — all inside
/// device_solver.cu, all out of scope here.
///
/// Logging goes through `cpu::pw::internal::log_iteration_energy` on purpose:
/// `core/log.h` is CUDA-free and must not be included from a `.cu`.

#include "atomiverse/cuda/planewave/pw_solver.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"
#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"

#include <array>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {

void PW::validate_members() const {
    // Exactly two functionals have device kernels: XCType::LSDA
    // (cuda/planewave/internal/xc_lda.cu, Slater + VWN5) and XCType::PBE
    // (internal/xc_pbe.cu, PBE-X + PW92-based PBE-C with the spectral GGA
    // passes in DeviceDensity). An explicit whitelist rather than
    // `!is_gga(xc)` keeps the gate honest for the PBE-family variants
    // (PBEsol/revPBE/RPBE) whose constants the device kernel does NOT carry.
    if (xc != cpu::scf::XCType::LSDA && xc != cpu::scf::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::PW::run: xc has no device path — only XCType::LSDA "
            "(Slater + VWN5) and XCType::PBE are implemented on the GPU. Set "
            "pw.xc to one of those, or run this functional on cpu::pw::PW.");
    if (spin_polarized)
        throw std::invalid_argument(
            "cuda::pw::PW::run: spin_polarized=true has no device path — the "
            "device density accumulator is fixed at nspin=1 with 2 electrons "
            "per band. Use cpu::pw::PW for collinear spin.");
    if (multiplicity != 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: multiplicity != 1 requires spin_polarized, "
            "which has no device path. Use cpu::pw::PW.");
    if (smearing != Smearing::none)
        throw std::invalid_argument(
            "cuda::pw::PW::run: smearing has no device path — occupations are "
            "fixed closed-shell on the GPU (degauss / mp_order / extra_bands "
            "are inert here). Use cpu::pw::PW for metallic smearing.");
    // k_mesh = 0 is the dangerous one: generate_mp_mesh loops zero times and
    // returns an empty grid without throwing, so run() would iterate over a
    // density that is never accumulated and report a converged e_band of 0.
    if (k_mesh < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: k_mesh must be >= 1 (Monkhorst-Pack mesh "
            "size per dimension; 1 = Γ-point only)");
    if (!(ecut_ry > 0.0))
        throw std::invalid_argument(
            "cuda::pw::PW::run: ecut_ry must be > 0 [Ry]");
    if (!(tol > 0.0))
        throw std::invalid_argument("cuda::pw::PW::run: tol must be > 0");
    if (!(mix_beta > 0.0))
        throw std::invalid_argument("cuda::pw::PW::run: mix_beta must be > 0");
    if (pulay_depth < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: pulay_depth must be >= 1");
    if (max_iter < 1)
        throw std::invalid_argument("cuda::pw::PW::run: max_iter must be >= 1");
}

void PW::build(const cpu::Frame& reference) {
    // Same envelope as cpu::pw::PW: exactly one pseudopotential source,
    // validated up front so a bad file or unknown GTH name fails here.
    if (upf_path.empty() == gth_name.empty())
        throw std::invalid_argument(
            "cuda::pw::PW::build: set exactly one of upf_path (a "
            "norm-conserving UPF file) or gth_name (an in-tree analytic GTH "
            "parameterization such as \"GTH-PBE-q4\"). A plane-wave run has "
            "no default pseudopotential.");
    if (!upf_path.empty()) {
        (void)cpu::pw::internal::read_upf_v1(upf_path);
    } else {
        const int* sp_type = reference.span(kAtom_type);
        if (reference.nlocal() > 0)
            (void)cpu::gpw::GthPotential(
                static_cast<molrs::Element>(sp_type[0]), gth_name);
    }
    built_ = true;
}

PWResult& PW::run(const cpu::Frame& frame) {
    // Reject unsupported members before any allocation or CUDA call.
    validate_members();

    // Lazy build — convenience for callers that skip explicit build().
    if (!built_)
        build(frame);

    result_ = PWResult{};

    // ── Shared problem assembly (identical to the CPU backend) ──
    // PWProblem is non-copyable (ws.gvec aliases gv0): construct in place.
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry,
                                        k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;

    if (prob.n_elec % 2 != 0)
        throw std::invalid_argument(
            "cuda::pw::PW::run: " + std::to_string(prob.n_elec)
            + " valence electrons is not closed-shell — the device band "
              "solver occupies every band with exactly 2 electrons. Use "
              "cpu::pw::PW with smearing or spin_polarized.");

    // Mixing members are the method's, not the workspace defaults. The device
    // mixer reads them through DeviceDensity's constructor; the workspace
    // fields are kept in sync so the host `ws` never lies about the run.
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    // build_kpoints() leaves ws bound to the density sphere, so ws.omega is
    // stable for the whole run; the k-loop no longer touches ws at all.
    const AccReal omega_bohr = ws.omega;

    // No MPI k-point distribution on this backend: every rank owns every k.
    result_.n_kpts = n_kpts;
    result_.n_kpts_local = n_kpts;

    // Device state: one-time H2D of the geometry-constant G-sphere tables and
    // the atomic ρ(G) seed, then nothing scientific leaves the GPU. Density
    // owns the non-default SCF stream; the band solver borrows it so both
    // halves stay ordered without events (cuda-pw-nondefault-stream).
    internal::DeviceDensity den(prob, mix_beta, pulay_depth, xc);
    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                                      den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    // Every per-k table (G→FFT map, T(G), q = k+G, KB descriptors, initial
    // Davidson subspace) is geometry — built and uploaded once, here.
    solver.build_kpoints(prob);

    // Same convergence policy object the CPU backend uses (warmup = 2); the
    // device reduces ‖ρ − ρ_old‖₂ itself, so feed the norm overload.
    cpu::pw::internal::ConvergenceCheck conv{AccReal(tol), 2};

    for (int iter = 0; iter < max_iter; ++iter) {
        den.hartree();              // PoissonSolver::step
        den.exchange_correlation(); // XCBuilder::step(ws, xc)
        den.assemble_veff();        // VeffBuilder::step
        den.zero_accumulator();     // DensityBuilder::begin

        AccReal e_band = 0;
        for (int ik = 0; ik < n_kpts; ++ik) {
            // The per-k workspace rebinding that used to live here — G-sphere,
            // KB/atomic-wfc table copies, the T(G) refill — was rebuilding
            // geometry that cannot change inside the SCF. It now happens once
            // in build_kpoints(); binding is a pointer swap.
            solver.bind_kpoint(ik);
            solver.solve(ws, 40);

            const AccReal weight = prob.kg.weights[ik];
            // Eigenvalues are the only band-side quantity the host needs;
            // they are already in ws (the Davidson subspace solve is a host
            // step). No wavefunction or density array is read back.
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);

            solver.accumulate_density(weight, omega_bohr);
        }

        den.collect();  // DensityBuilder::finish, first half
        den.mix(iter);  // pulay_mix (host solve of the ≤ 8×8 DIIS system)
        const auto s = den.reduce(static_cast<double>(vol_bohr));

        // The shared logger reads ws.E_xc; every other host density-side
        // array is stale after W2 and is deliberately never read again.
        ws.E_xc = AccReal(s.e_xc);

        // The SHARED decomposition. `iteration_energy` itself cannot be
        // called: its Parseval G-loop reads ws.rho_G / ws.V_*_G, which after
        // W2 live only on the device. Its combiner half can, and does — one
        // definition of e_one / total for both backends, so the
        // catastrophic-cancellation term order cannot drift apart.
        const auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            ws.E_xc, prob.ew_total, ws.E_aug);

        cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total, iter);

        result_.iterations = iter + 1;
        result_.energy_ha = static_cast<double>(terms.total);
        result_.energy =
            result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;

        if (conv.step(AccReal(s.resid))) {
            result_.converged = true;
            break;
        }
    }

    return result_;
}

std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PW& method) {
    // Same gate as run(): reject unsupported members before any CUDA work.
    method.validate_members();
    if (frame.nlocal() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: frame has no atoms");

    // Lazy build on a non-const view of the same public members. build() only
    // sets built_ and validates the PP source — method is const, so replicate
    // the PP gate without mutating the caller's object.
    if (method.upf_path.empty() == method.gth_name.empty())
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: set exactly one of upf_path or "
            "gth_name (same envelope as PW::build)");
    if (!method.upf_path.empty()) {
        (void)cpu::pw::internal::read_upf_v1(method.upf_path);
    } else {
        const int* sp_type = frame.span(kAtom_type);
        if (frame.nlocal() > 0)
            (void)cpu::gpw::GthPotential(
                static_cast<molrs::Element>(sp_type[0]), method.gth_name);
    }

    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, method.upf_path, method.gth_name,
                                        method.ecut_ry, method.k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;

    if (prob.n_elec % 2 != 0)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: "
            + std::to_string(prob.n_elec)
            + " valence electrons is not closed-shell — use cpu::pw for "
              "open-shell / metallic paths");

    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    const AccReal omega_bohr = ws.omega;

    // Same stream coupling as PW::run: density owns the SCF stream; the band
    // solver borrows it so V_eff / ρ_acc writes stay ordered without events.
    // Creating a second stream here races the two halves and breaks PBE SCF
    // (many more async FFTs on the density side than LSDA).
    internal::DeviceDensity den(prob, method.mix_beta, method.pulay_depth,
                                method.xc);
    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                                      den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);

    cpu::pw::internal::ConvergenceCheck conv{AccReal(method.tol), 2};
    AccReal last_energy_ha = AccReal(0);
    int last_iter = 0;
    for (int iter = 0; iter < method.max_iter; ++iter) {
        den.hartree();
        den.exchange_correlation();
        den.assemble_veff();
        den.zero_accumulator();

        AccReal e_band = 0;
        for (int ik = 0; ik < n_kpts; ++ik) {
            solver.bind_kpoint(ik);
            solver.solve(ws, 40);
            const AccReal weight = prob.kg.weights[ik];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);
            solver.accumulate_density(weight, omega_bohr);
        }

        den.collect();
        den.mix(iter);
        const auto s = den.reduce(static_cast<double>(vol_bohr));
        ws.E_xc = AccReal(s.e_xc);
        const auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            ws.E_xc, prob.ew_total, ws.E_aug);
        last_energy_ha = terms.total;
        last_iter = iter + 1;
        if (conv.step(AccReal(s.resid)))
            break;
    }
    // Host-visible breadcrumb: energy of the force-path SCF (Ry). Mirrors the
    // CPU `forces-scf converged` line so dual-gate logs can compare. Do not
    // include core/log.h from .cu (see file header); printf is enough.
    std::printf("[pw] cuda forces-scf done iter=%d energy_ry=%.8f\n", last_iter,
                static_cast<double>(last_energy_ha)
                    / cpu::pw::qe_units::rydberg_to_hartree);

    // Forces from the final mixed density + band re-solve at final V_eff.
    return internal::pw_forces_analytic_impl(frame, method.upf_path,
                                             method.gth_name, method.ecut_ry,
                                             den, solver, prob, ws);
}

std::array<double, 6> pw_stress(const cpu::Frame& frame, const PW& method,
                                double* energy_ry_out) {
    method.validate_members();
    if (frame.nlocal() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: non-positive cell volume");

    if (method.upf_path.empty() == method.gth_name.empty())
        throw std::invalid_argument(
            "cuda::pw::pw_stress: set exactly one of upf_path or gth_name "
            "(same envelope as PW::build)");
    if (!method.upf_path.empty()) {
        (void)cpu::pw::internal::read_upf_v1(method.upf_path);
    } else {
        const int* sp_type = frame.span(kAtom_type);
        if (frame.nlocal() > 0)
            (void)cpu::gpw::GthPotential(
                static_cast<molrs::Element>(sp_type[0]), method.gth_name);
    }

    return internal::pw_stress_analytic_impl(
        frame, method.upf_path, method.gth_name, method.ecut_ry, method.k_mesh,
        method.tol, method.max_iter, method.mix_beta, method.pulay_depth,
        method.xc, energy_ry_out);
}

} // namespace pw
} // namespace cuda
} // namespace ATV
