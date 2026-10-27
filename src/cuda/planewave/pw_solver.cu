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
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"
#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"
#include "atomiverse/electronic/smearing.h"

#include <algorithm>
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
    if (xc != cpu::xc::XCType::LSDA && xc != cpu::xc::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::PW::run: xc has no device path — only XCType::LSDA "
            "(Slater + VWN5) and XCType::PBE are implemented on the GPU. "
            "Construct with one of those, or run this functional on "
            "cpu::pw::PW.");
    // Collinear UKS: LSDA and true spin-GGA PBE are landed; other XC throw.
    if (spin_polarized && xc != cpu::xc::XCType::LSDA
        && xc != cpu::xc::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::PW::run: spin_polarized supports XCType::LSDA and "
            "XCType::PBE only on the device path — set pw.xc to one of those, "
            "or use cpu::pw::PW for other spin functionals.");
    if (spin_polarized && multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: multiplicity must be >= 1 when "
            "spin_polarized");
    if (!spin_polarized && multiplicity != 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: multiplicity != 1 requires spin_polarized. "
            "Set spin_polarized=true for collinear UKS, or multiplicity=1.");
    if (smearing != ATV::electronic::Smearing::none && !(degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::PW::run: degauss must be > 0 [Ry] when smearing is "
            "enabled");
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
    cpu::pw::internal::validate_potential_frame(reference, *potential);
    built_ = true;
}

PWResult& PW::run(const cpu::Frame& frame) {
    // Reject unsupported members before any allocation or CUDA call.
    validate_members();
    cpu::pw::internal::validate_potential_frame(frame, *potential);

    // Lazy build — convenience for callers that skip explicit build().
    if (!built_)
        build(frame);

    // ── Shared problem assembly (identical to the CPU backend) ──
    // PWProblem is non-copyable (ws.gvec aliases gv0): construct in place.
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, *potential, ecut_ry, k_mesh,
                                        prob);
    result_ = PWResult{};
    cpu::pw::internal::PWWorkspace& ws = prob.ws;

    // Collinear occupation split: N↑ − N↓ = multiplicity − 1, N↑ + N↓ = n_elec.
    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != ATV::electronic::Smearing::none);
    int n_alpha = n_elec / 2;
    int n_beta = n_elec - n_alpha;
    if (spin_polarized) {
        const int mag = multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "cuda::pw::PW::run: n_elec and multiplicity have incompatible "
                "parity (need N↑,N↓ non-negative integers)");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
        result_.magnetization = static_cast<double>(mag);
    } else {
        // Fixed closed-shell occupations require even n_elec. Metallic smear
        // fills fractional occupations and accepts odd valence counts (Al).
        if (!use_smear && n_elec % 2 != 0)
            throw std::invalid_argument(
                "cuda::pw::PW::run: " + std::to_string(n_elec)
                + " valence electrons is not closed-shell — the restricted "
                  "fixed-occupation path occupies every band with exactly 2 "
                  "electrons. Use spin_polarized for open shell, or set "
                  "smearing for metallic occupations.");
        result_.magnetization = 0;
    }

    // Mixing members are the method's, not the workspace defaults. The device
    // mixer reads them through DeviceDensity's constructor; the workspace
    // fields are kept in sync so the host `ws` never lies about the run.
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    // Band count: fixed occ uses n_elec/2 (or max spin channel); smear uses
    // ceil(n_e/2) + extra_bands so empty states sit near E_F for the μ search.
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max({n_bands, n_occ_ceil + std::max(extra_bands, 0), 1});
    }
    const AccReal vol_bohr = prob.vol_bohr;
    // build_kpoints() leaves ws bound to the density sphere, so ws.omega is
    // stable for the whole run; the k-loop no longer touches ws at all.
    const AccReal omega_bohr = ws.omega;

    ATV::electronic::Smearing smear_kind =
        ATV::electronic::Smearing::fermi_dirac;
    if (smearing == ATV::electronic::Smearing::fermi_dirac)
        smear_kind = ATV::electronic::Smearing::fermi_dirac;
    else if (smearing == ATV::electronic::Smearing::methfessel_paxton)
        smear_kind = ATV::electronic::Smearing::methfessel_paxton;
    else if (smearing == ATV::electronic::Smearing::gaussian)
        smear_kind = ATV::electronic::Smearing::gaussian;
    const AccReal degauss_ha =
        AccReal(degauss) * AccReal(cpu::pw::qe_units::rydberg_to_hartree);

    // No MPI k-point distribution on this backend: every rank owns every k.
    result_.n_kpts = n_kpts;
    result_.n_kpts_local = n_kpts;

    // Device state: one-time H2D of the geometry-constant G-sphere tables and
    // the atomic ρ(G) seed, then nothing scientific leaves the GPU. Density
    // owns the non-default SCF stream; the band solver borrows it so both
    // halves stay ordered without events (cuda-pw-nondefault-stream).
    internal::DeviceDensity den(prob, mix_beta, pulay_depth, xc, spin_polarized,
                                n_alpha, n_beta, n_elec);
    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                                      den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    // Every per-k table (G→FFT map, T(G), q = k+G, KB descriptors, initial
    // Davidson subspace) is geometry — built and uploaded once, here.
    solver.build_kpoints(prob);
    if (use_smear)
        solver.ensure_ritz_cache();

    // Same convergence policy object the CPU backend uses (warmup = 2); the
    // device reduces ‖ρ − ρ_old‖₂ itself, so feed the norm overload.
    cpu::pw::internal::ConvergenceCheck conv{AccReal(tol), 2};

    if (spin_polarized && use_smear) {
        // Collinear UKS + metallic smear: shared μ over spin-orbitals,
        // FD/Gaussian probabilities or signed MP quasi-occupations (CPU twin).
        // Ritz cache is per-k only,
        // so pass 2 re-solves after μ (Γ dual-gate cost; not a new kernel).
        const int n_bs = n_bands;
        const int mp_order = this->mp_order;
        std::vector<std::vector<AccReal>> evals_sk(
            static_cast<std::size_t>(2 * n_kpts));
        for (int iter = 0; iter < max_iter; ++iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();

            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    ev.assign(ws.eigenvalues.begin(), ws.eigenvalues.end());
                    if (static_cast<int>(ev.size()) > n_bs)
                        ev.resize(static_cast<std::size_t>(n_bs));
                }
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(2 * n_kpts * n_bs));
            all_weights.reserve(static_cast<std::size_t>(2 * n_kpts * n_bs));
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    for (int b = 0; b < n_bs && b < static_cast<int>(ev.size());
                         ++b) {
                        all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                        all_weights.push_back(w);
                    }
                }
            }
            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order, AccReal(1));

            AccReal e_band = 0;
            AccReal entropy_sum = 0;
            AccReal n_up = 0;
            AccReal n_dn = 0;
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    std::vector<AccReal> w_one(ev.size(), w);
                    auto fill = ATV::electronic::fill_occupations(
                        ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order,
                        AccReal(1));
                    entropy_sum += fill.entropy;
                    AccReal n_s = 0;
                    const int nb = static_cast<int>(fill.occ.size());
                    for (int b = 0; b < nb; ++b) {
                        e_band += w * ev[static_cast<std::size_t>(b)]
                                  * fill.occ[static_cast<std::size_t>(b)];
                        n_s += w * fill.occ[static_cast<std::size_t>(b)];
                    }
                    if (s == 0)
                        n_up += n_s;
                    else
                        n_dn += n_s;
                    solver.accumulate_density(w, omega_bohr, fill.occ.data(),
                                              nb);
                }
                den.collect_spin_out(s);
            }

            den.mix_spin_linear();
            const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            const AccReal e_ent_ha = -degauss_ha * entropy_sum;
            terms.total = terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total
                          + e_ent_ha;
            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);
            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(terms.total);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;
            result_.magnetization = static_cast<double>(n_up - n_dn);
            result_.n_elec_occ = static_cast<double>(n_up + n_dn);
            result_.e_fermi = static_cast<double>(mu_ha)
                              / cpu::pw::qe_units::rydberg_to_hartree;
            result_.e_entropy = static_cast<double>(e_ent_ha)
                                / cpu::pw::qe_units::rydberg_to_hartree;
            if (conv.step(AccReal(s.resid))) {
                result_.converged = true;
                break;
            }
        }
        return result_;
    }

    if (spin_polarized) {
        // ── Collinear fixed-occupation UKS (CPU twin in pw_solver.cpp) ──
        const int n_occ[2] = {n_alpha, n_beta};
        for (int iter = 0; iter < max_iter; ++iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();

            AccReal e_band = 0;
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal weight = prob.kg.weights[ik];
                    for (int b = 0; b < n_occ[s]; ++b)
                        e_band += weight * ws.eigenvalues[b]; // spin-orbital
                    // f = 1 per occupied spin-orbital (CPU accumulate_kpoint).
                    solver.accumulate_density(weight, omega_bohr, n_occ[s],
                                              1.0);
                }
                den.collect_spin_out(s);
            }

            den.mix_spin_linear();
            const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);

            // Spin energy: e_one = e_band − 2 e_h − ∫(ρ↑V↑+ρ↓V↓).
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            terms.total =
                terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total;

            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);

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

    if (use_smear) {
        // ── Restricted metallic-smearing SCF (CPU twin in pw_solver.cpp) ──
        // Pass 1: diagonalize every k, stash evals (host) + Ritz (device).
        // μ from the full spectrum; Pass 2: fractional density accumulate.
        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
        for (int iter = 0; iter < max_iter; ++iter) {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();

            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                evals_k[static_cast<std::size_t>(ik)].assign(
                    ws.eigenvalues.begin(), ws.eigenvalues.end());
                if (static_cast<int>(
                        evals_k[static_cast<std::size_t>(ik)].size())
                    > n_bands)
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_bands));
                solver.stash_ritz();
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                     ++b) {
                    all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                    all_weights.push_back(w);
                }
            }

            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order);

            AccReal n_occ = AccReal(0);
            AccReal entropy_sum = AccReal(0);
            std::vector<std::vector<AccReal>> occ_k(
                static_cast<std::size_t>(n_kpts));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                std::vector<AccReal> w_one(ev.size(), w);
                auto fill = ATV::electronic::fill_occupations(
                    ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
                occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
                for (std::size_t b = 0; b < ev.size(); ++b)
                    occ_k[static_cast<std::size_t>(ik)][b] =
                        ATV::electronic::smear_f_band(ev[b], mu_ha, degauss_ha,
                                                      smear_kind, mp_order);
                n_occ += fill.n_occ;
                entropy_sum += fill.entropy;
            }

            den.zero_accumulator();
            AccReal e_band = 0;
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.load_ritz(ik);
                const AccReal weight =
                    prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& occ = occ_k[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                const int nb = static_cast<int>(occ.size());
                for (int b = 0; b < nb; ++b)
                    e_band += weight * ev[static_cast<std::size_t>(b)]
                              * occ[static_cast<std::size_t>(b)];
                solver.accumulate_density(weight, omega_bohr, occ.data(), nb);
            }

            den.collect();
            den.mix(iter);
            const auto s = den.reduce(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);

            const auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            // Mermin free energy: F = E_KS − σ · S  (σ = degauss in Ha).
            const AccReal e_ent_ha = -degauss_ha * entropy_sum;
            const AccReal e_ha = terms.total + e_ent_ha;

            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);

            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(e_ha);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;
            result_.e_fermi = static_cast<double>(mu_ha)
                              / cpu::pw::qe_units::rydberg_to_hartree;
            result_.n_elec_occ = static_cast<double>(n_occ);
            result_.e_entropy = static_cast<double>(e_ent_ha)
                                / cpu::pw::qe_units::rydberg_to_hartree;

            if (conv.step(AccReal(s.resid))) {
                result_.converged = true;
                break;
            }
        }
        return result_;
    }

    // ── Restricted fixed closed-shell occupations ──
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

        den.collect(); // DensityBuilder::finish, first half
        den.mix(iter); // pulay_mix (host solve of the ≤ 8×8 DIIS system)
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
            ws.E_xc, prob.ew_total);

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

PWResult& PW::run(cpu::Frame& frame) {
    PWResult& res = run(std::as_const(frame));
    frame.meta.get(kEnergy) = res.energy * cpu::pw::qe_units::rydberg_to_ev;
    return res;
}

PWResult& PW::forces(cpu::Frame& frame) {
    if (!built_)
        build(frame);
    const std::vector<std::array<double, 3>> F =
        forces_impl(frame, nullptr, &result_);
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i) {
        const auto& f = F[static_cast<std::size_t>(i)];
        fx[i] = static_cast<AccReal>(f[0]);
        fy[i] = static_cast<AccReal>(f[1]);
        fz[i] = static_cast<AccReal>(f[2]);
    }
    frame.meta.get(kEnergy) = result_.energy * cpu::pw::qe_units::rydberg_to_ev;
    return result_;
}

std::array<double, 6> PW::stress(cpu::Frame& frame) {
    if (!built_)
        build(frame);
    const std::array<double, 6> sigma = stress_impl(frame, nullptr, &result_);
    const double vol = frame.box.volume();
    auto& st = frame.meta.get(kStress);
    for (int k = 0; k < 6; ++k)
        st[static_cast<std::size_t>(k)] =
            sigma[static_cast<std::size_t>(k)] * vol;
    frame.meta.get(kEnergy) = result_.energy * cpu::pw::qe_units::rydberg_to_ev;
    return sigma;
}

std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PW& method,
                                                      double* energy_ha_out) {
    return method.forces_impl(frame, energy_ha_out, nullptr);
}

std::vector<std::array<double, 3>> PW::forces_impl(const cpu::Frame& frame,
                                                   double* energy_ha_out,
                                                   PWResult* result_out) const {
    const PW& method = *this;
    PWResult force_result{};
    if (result_out)
        *result_out = PWResult{};
    // Gate unsupported force surface before any CUDA work. Restricted fixed
    // + restricted metallic smear + collinear LSDA/PBE fixed-occ + collinear
    // spin×smear forces with a shared μ and signed MP quasi-occupations are
    // supported. Stress uses the
    // same occupation convention in its strained NSCF evaluations.
    method.validate_members();
    if (method.spin_polarized && method.xc != cpu::xc::XCType::LSDA
        && method.xc != cpu::xc::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: spin_polarized force path is "
            "XCType::LSDA or XCType::PBE only — use "
            "cpu::pw::pw_forces_analytic for other XC");
    const bool use_smear = (method.smearing != ATV::electronic::Smearing::none);
    if (use_smear && !(method.degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: degauss must be > 0 [Ry] when "
            "smearing is enabled");
    if (method.spin_polarized && method.multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: multiplicity must be >= 1 when "
            "spin_polarized");
    if (frame.nlocal() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: frame has no atoms");

    cpu::pw::internal::validate_potential_frame(frame, *method.potential);

    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, *method.potential,
                                        method.ecut_ry, method.k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;

    const int n_elec = prob.n_elec;
    int n_alpha = -1;
    int n_beta = -1;
    if (method.spin_polarized) {
        const int mag = method.multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "cuda::pw::pw_forces_analytic: n_elec and multiplicity have "
                "incompatible parity (need N↑,N↓ non-negative integers)");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
    } else if (!use_smear && n_elec % 2 != 0) {
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: " + std::to_string(n_elec)
            + " valence electrons is not closed-shell — use spin_polarized "
              "for open shell, or set smearing for metallic occupations");
    }

    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    force_result.n_kpts = n_kpts;
    force_result.n_kpts_local = n_kpts;
    force_result.magnetization =
        method.spin_polarized ? method.multiplicity - 1 : 0;
    int n_bands = method.spin_polarized ? std::max(std::max(n_alpha, n_beta), 1)
                                        : prob.n_bands;
    ATV::electronic::Smearing smear_kind =
        ATV::electronic::Smearing::fermi_dirac;
    AccReal degauss_ha = AccReal(0);
    if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max(
            {n_bands, n_occ_ceil + std::max(method.extra_bands, 0), 1});
        if (method.smearing == ATV::electronic::Smearing::methfessel_paxton)
            smear_kind = ATV::electronic::Smearing::methfessel_paxton;
        else if (method.smearing == ATV::electronic::Smearing::gaussian)
            smear_kind = ATV::electronic::Smearing::gaussian;
        degauss_ha = AccReal(method.degauss)
                     * AccReal(cpu::pw::qe_units::rydberg_to_hartree);
    }
    const AccReal vol_bohr = prob.vol_bohr;
    const AccReal omega_bohr = ws.omega;

    // Same stream coupling as PW::run: density owns the SCF stream; the band
    // solver borrows it so V_eff / ρ_acc writes stay ordered without events.
    internal::DeviceDensity den(prob, method.mix_beta, method.pulay_depth,
                                method.xc, method.spin_polarized, n_alpha,
                                n_beta, n_elec);
    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                                      den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);

    cpu::pw::internal::ConvergenceCheck conv{AccReal(method.tol), 2};
    AccReal last_energy_ha = AccReal(0);
    int last_iter = 0;

    if (method.spin_polarized && use_smear) {
        // Same SCF as PW::run: shared μ over both spin spectra, preserving
        // signed MP quasi-occupations as well as FD/Gaussian probabilities.
        // Must not fall through to the fixed-occ UKS branch (that SCF
        // energy is the spin-PBE closed-shell value, not the Mermin F).
        const int n_bs = n_bands;
        const int mp_order = method.mp_order;
        std::vector<std::vector<AccReal>> evals_sk(
            static_cast<std::size_t>(2 * n_kpts));
        for (int iter = 0; iter < method.max_iter; ++iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    ev.assign(ws.eigenvalues.begin(), ws.eigenvalues.end());
                    if (static_cast<int>(ev.size()) > n_bs)
                        ev.resize(static_cast<std::size_t>(n_bs));
                }
            }
            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    for (int b = 0; b < n_bs && b < static_cast<int>(ev.size());
                         ++b) {
                        all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                        all_weights.push_back(w);
                    }
                }
            }
            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order, AccReal(1));
            AccReal e_band = 0;
            AccReal entropy_sum = 0;
            AccReal spin_count[2] = {0, 0};
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    std::vector<AccReal> w_one(ev.size(), w);
                    auto fill = ATV::electronic::fill_occupations(
                        ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order,
                        AccReal(1));
                    entropy_sum += fill.entropy;
                    spin_count[s] += fill.n_occ;
                    const int nb = static_cast<int>(fill.occ.size());
                    for (int b = 0; b < nb; ++b)
                        e_band += w * ev[static_cast<std::size_t>(b)]
                                  * fill.occ[static_cast<std::size_t>(b)];
                    solver.accumulate_density(w, omega_bohr, fill.occ.data(),
                                              nb);
                }
                den.collect_spin_out(s);
            }
            den.mix_spin_linear();
            const auto red = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(red.e_xc);
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(red.e_h), AccReal(red.e_xc_int),
                AccReal(red.e_loc), ws.E_xc, prob.ew_total);
            terms.e_one = e_band - AccReal(2.0) * AccReal(red.e_h)
                          - AccReal(red.e_xc_int);
            last_energy_ha = terms.e_one + AccReal(red.e_h) + ws.E_xc
                             + prob.ew_total - degauss_ha * entropy_sum;
            force_result.magnetization =
                static_cast<double>(spin_count[0] - spin_count[1]);
            force_result.n_elec_occ =
                static_cast<double>(spin_count[0] + spin_count[1]);
            force_result.e_fermi = static_cast<double>(mu_ha)
                                   / cpu::pw::qe_units::rydberg_to_hartree;
            force_result.e_entropy =
                static_cast<double>(-degauss_ha * entropy_sum)
                / cpu::pw::qe_units::rydberg_to_hartree;
            last_iter = iter + 1;
            if (conv.step(AccReal(red.resid))) {
                force_result.converged = true;
                break;
            }
        }
    } else if (method.spin_polarized) {
        // Collinear fixed-occupation UKS force-path SCF (CPU twin).
        const int n_occ[2] = {n_alpha, n_beta};
        for (int iter = 0; iter < method.max_iter; ++iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();

            AccReal e_band = 0;
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal weight = prob.kg.weights[ik];
                    for (int b = 0; b < n_occ[s]; ++b)
                        e_band += weight * ws.eigenvalues[b];
                    solver.accumulate_density(weight, omega_bohr, n_occ[s],
                                              1.0);
                }
                den.collect_spin_out(s);
            }

            den.mix_spin_linear();
            const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            terms.total =
                terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total;
            last_energy_ha = terms.total;
            last_iter = iter + 1;
            if (conv.step(AccReal(s.resid))) {
                force_result.converged = true;
                break;
            }
        }
    } else if (use_smear) {
        // Restricted metallic-smear force-path SCF (CPU twin via
        // run_scf_smear_restricted / energy path in PW::run).
        const int mp_order = method.mp_order;
        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
        for (int iter = 0; iter < method.max_iter; ++iter) {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();

            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                evals_k[static_cast<std::size_t>(ik)].assign(
                    ws.eigenvalues.begin(), ws.eigenvalues.end());
                if (static_cast<int>(
                        evals_k[static_cast<std::size_t>(ik)].size())
                    > n_bands)
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_bands));
                solver.stash_ritz();
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                     ++b) {
                    all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                    all_weights.push_back(w);
                }
            }

            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order);

            AccReal e_band = 0;
            AccReal entropy_sum = AccReal(0);
            AccReal n_occ = 0;
            std::vector<std::vector<AccReal>> occ_k(
                static_cast<std::size_t>(n_kpts));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                std::vector<AccReal> w_one(ev.size(), w);
                auto fill = ATV::electronic::fill_occupations(
                    ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
                occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
                for (std::size_t b = 0; b < ev.size(); ++b)
                    occ_k[static_cast<std::size_t>(ik)][b] =
                        ATV::electronic::smear_f_band(ev[b], mu_ha, degauss_ha,
                                                      smear_kind, mp_order);
                entropy_sum += fill.entropy;
                n_occ += fill.n_occ;
            }

            den.zero_accumulator();
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.load_ritz(ik);
                const AccReal weight =
                    prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& occ = occ_k[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                const int nb = static_cast<int>(occ.size());
                for (int b = 0; b < nb; ++b)
                    e_band += weight * ev[static_cast<std::size_t>(b)]
                              * occ[static_cast<std::size_t>(b)];
                solver.accumulate_density(weight, omega_bohr, occ.data(), nb);
            }

            den.collect();
            den.mix(iter);
            const auto s = den.reduce(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);
            const auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            const AccReal e_ent_ha = -degauss_ha * entropy_sum;
            last_energy_ha = terms.total + e_ent_ha;
            force_result.n_elec_occ = static_cast<double>(n_occ);
            force_result.e_fermi = static_cast<double>(mu_ha)
                                   / cpu::pw::qe_units::rydberg_to_hartree;
            force_result.e_entropy = static_cast<double>(e_ent_ha)
                                     / cpu::pw::qe_units::rydberg_to_hartree;
            last_iter = iter + 1;
            if (conv.step(AccReal(s.resid))) {
                force_result.converged = true;
                break;
            }
        }
    } else {
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
                ws.E_xc, prob.ew_total);
            last_energy_ha = terms.total;
            last_iter = iter + 1;
            if (conv.step(AccReal(s.resid))) {
                force_result.converged = true;
                break;
            }
        }
    }
    // Host-visible breadcrumb: energy of the force-path SCF (Ry). Mirrors the
    // CPU `forces-scf converged` line so dual-gate logs can compare.
    std::printf(
        "[pw] cuda forces-scf done iter=%d energy_ry=%.8f%s%s\n", last_iter,
        static_cast<double>(last_energy_ha)
            / cpu::pw::qe_units::rydberg_to_hartree,
        method.spin_polarized ? " spin=1" : "", use_smear ? " smear=1" : "");
    if (energy_ha_out != nullptr)
        *energy_ha_out = static_cast<double>(last_energy_ha);

    force_result.iterations = last_iter;
    force_result.energy_ha = static_cast<double>(last_energy_ha);
    force_result.energy =
        force_result.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;
    // Forces from the final mixed density + band re-solve at final V_eff.
    auto forces = internal::pw_forces_analytic_impl(
        frame, den, solver, prob, ws, n_alpha, n_beta, use_smear, n_elec,
        smear_kind, static_cast<double>(degauss_ha), method.mp_order);
    if (result_out)
        *result_out = force_result;
    return forces;
}

std::array<double, 6> pw_stress(const cpu::Frame& frame, const PW& method,
                                double* energy_ry_out) {
    return method.stress_impl(frame, energy_ry_out, nullptr);
}

std::array<double, 6> PW::stress_impl(const cpu::Frame& frame,
                                      double* energy_ry_out,
                                      PWResult* result_out) const {
    const PW& method = *this;
    if (result_out)
        *result_out = PWResult{};
    method.validate_members();
    // Collinear LSDA/PBE stress supports fixed and fractional occupations.
    if (method.spin_polarized && method.xc != cpu::xc::XCType::LSDA
        && method.xc != cpu::xc::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: spin_polarized stress path is XCType::LSDA "
            "or XCType::PBE only — use cpu::pw::pw_stress for other XC");
    if (method.spin_polarized && method.multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: multiplicity must be >= 1 when "
            "spin_polarized");
    // Restricted and collinear smearing differentiate the same free energy:
    // both strained spin spectra are refilled at a shared chemical potential.
    if (method.smearing != ATV::electronic::Smearing::none
        && !(method.degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::pw_stress: degauss must be > 0 [Ry] when smearing is "
            "enabled");
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("cuda::pw::pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: non-positive cell volume");

    cpu::pw::internal::validate_potential_frame(frame, *method.potential);

    return internal::pw_stress_analytic_impl(
        frame, *method.potential, method.ecut_ry, method.k_mesh, method.tol,
        method.max_iter, method.mix_beta, method.pulay_depth, method.xc,
        method.smearing, method.degauss, method.mp_order, method.extra_bands,
        method.spin_polarized, method.multiplicity, energy_ry_out, result_out);
}

} // namespace pw
} // namespace cuda
} // namespace ATV
