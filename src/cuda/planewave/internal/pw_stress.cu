/// @file cuda/planewave/internal/pw_stress.cu
/// @brief Device frozen-density cell stress (one SCF + NSCF strain-FD).
///
/// Restricted/collinear LSDA or PBE, with fixed or fractional occupations.
/// Each strained spin-smearing spectrum is refilled at a shared chemical
/// potential and includes the same free-energy correction as its SCF.

#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/electronic/smearing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {
namespace {

void voigt_pair(int v, int& a, int& b) {
    static const int A[6] = {0, 1, 2, 0, 0, 1};
    static const int B[6] = {0, 1, 2, 1, 2, 2};
    a = A[v];
    b = B[v];
}

void apply_strain(cpu::Frame& frame, const Mat3& strain) {
    Mat3 F{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            F(a, b) = (a == b ? AccReal(1) : AccReal(0)) + strain(a, b);

    Mat3 cell = frame.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    frame.box.set_cell(cell_new);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        AccReal rx = x[a], ry = y[a], rz = z[a];
        x[a] = F(0, 0) * rx + F(0, 1) * ry + F(0, 2) * rz;
        y[a] = F(1, 0) * rx + F(1, 1) * ry + F(1, 2) * rz;
        z[a] = F(2, 0) * rx + F(2, 1) * ry + F(2, 2) * rz;
    }
}

ATV::electronic::Smearing to_smear_kind(ATV::electronic::Smearing s) {
    // Fixed occupations are not a law. Callers that still ask map onto
    // Fermi–Dirac, which is what this helper did before the enums merged.
    if (s == ATV::electronic::Smearing::methfessel_paxton)
        return ATV::electronic::Smearing::methfessel_paxton;
    if (s == ATV::electronic::Smearing::gaussian)
        return ATV::electronic::Smearing::gaussian;
    return ATV::electronic::Smearing::fermi_dirac;
}

/// Device SCF at `frame`; fills free/total energy and returns host ρ(r) + Ω.
/// Collinear spin: rho_r is unused; rho_a / rho_b hold the spin channels.
struct RefScf {
    // Stable ownership: ws.gvec aliases this problem's gv0. Strained samples
    // reuse its discrete Miller labels and FFT shape, never its stale host rho.
    std::shared_ptr<cpu::pw::internal::PWProblem> problem;
    std::vector<double> rho_r;
    std::vector<double> rho_a;
    std::vector<double> rho_b;
    double omega_bohr = 0;
    double energy_ry = 0;
    cpu::pw::PWResult result;
    int nr = 0;
    int iterations = 0;
    bool converged = false;
    bool spin = false;
    int n_alpha = 0;
    int n_beta = 0;
};

RefScf
run_device_scf(const cpu::Frame& frame,
               const ::ATV::electronic::NormConservingPotential& potential,
               double ecut_ry, int k_mesh, double tol, int max_iter,
               double mix_beta, int pulay_depth, cpu::xc::XCType xc,
               ATV::electronic::Smearing smearing, double degauss, int mp_order,
               int extra_bands, bool spin_polarized, int multiplicity) {
    auto owner = std::make_shared<cpu::pw::internal::PWProblem>();
    auto& prob = *owner;
    cpu::pw::internal::setup_pw_problem(frame, potential, ecut_ry, k_mesh,
                                        prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != ATV::electronic::Smearing::none);

    int n_alpha = 0;
    int n_beta = 0;
    if (spin_polarized) {
        const int mag = multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "cuda::pw::pw_stress: n_elec and multiplicity have "
                "incompatible parity");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
    } else if (!use_smear && n_elec % 2 != 0) {
        throw std::invalid_argument(
            "cuda::pw::pw_stress: " + std::to_string(n_elec)
            + " valence electrons is not closed-shell — use restricted "
              "metallic smearing for open-shell / metallic paths, or "
              "spin_polarized for collinear spin");
    }

    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max({n_bands, n_occ_ceil + std::max(extra_bands, 0), 1});
    }
    const AccReal vol_bohr = prob.vol_bohr;
    const AccReal omega_bohr = ws.omega;

    const ATV::electronic::Smearing smear_kind = to_smear_kind(smearing);
    const AccReal degauss_ha =
        AccReal(degauss) * AccReal(cpu::pw::qe_units::rydberg_to_hartree);

    DeviceDensity den(prob, mix_beta, pulay_depth, xc, spin_polarized, n_alpha,
                      n_beta, n_elec);
    DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                            den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);
    if (use_smear && !spin_polarized)
        solver.ensure_ritz_cache();

    cpu::pw::internal::ConvergenceCheck conv{AccReal(tol), 2};
    AccReal last_energy_ha = AccReal(0);
    int last_iter = 0;
    bool converged = false;
    cpu::pw::PWResult result;
    result.n_kpts = n_kpts;
    result.n_kpts_local = n_kpts;
    result.magnetization = spin_polarized ? multiplicity - 1 : 0;

    if (spin_polarized && use_smear) {
        for (int iter = 0; iter < max_iter; ++iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
            std::vector<std::vector<AccReal>> evals_sk(
                static_cast<std::size_t>(2 * n_kpts));
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    evals_sk[static_cast<std::size_t>(s * n_kpts + ik)].assign(
                        ws.eigenvalues.begin(), ws.eigenvalues.end());
                }
            }
            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    for (AccReal e :
                         evals_sk[static_cast<std::size_t>(s * n_kpts + ik)]) {
                        all_evals.push_back(e);
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
            result.magnetization =
                static_cast<double>(spin_count[0] - spin_count[1]);
            result.n_elec_occ =
                static_cast<double>(spin_count[0] + spin_count[1]);
            result.e_fermi = static_cast<double>(mu_ha)
                             / cpu::pw::qe_units::rydberg_to_hartree;
            result.e_entropy = static_cast<double>(-degauss_ha * entropy_sum)
                               / cpu::pw::qe_units::rydberg_to_hartree;
            last_iter = iter + 1;
            if (conv.step(AccReal(red.resid))) {
                converged = true;
                break;
            }
        }
    } else if (spin_polarized) {
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
                converged = true;
                break;
            }
        }
    } else if (use_smear) {
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
            last_energy_ha = terms.total + (-degauss_ha * entropy_sum);
            result.n_elec_occ = static_cast<double>(n_occ);
            result.e_fermi = static_cast<double>(mu_ha)
                             / cpu::pw::qe_units::rydberg_to_hartree;
            result.e_entropy = static_cast<double>(-degauss_ha * entropy_sum)
                               / cpu::pw::qe_units::rydberg_to_hartree;
            last_iter = iter + 1;
            if (conv.step(AccReal(s.resid))) {
                converged = true;
                break;
            }
        }
    } else {
        for (int iter = 0; iter < max_iter; ++iter) {
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
                converged = true;
                break;
            }
        }
    }

    RefScf out;
    out.problem = owner;
    out.nr = den.nr();
    out.omega_bohr = den.omega();
    out.spin = spin_polarized;
    out.n_alpha = n_alpha;
    out.n_beta = n_beta;
    if (spin_polarized) {
        out.rho_a.resize(static_cast<std::size_t>(out.nr));
        out.rho_b.resize(static_cast<std::size_t>(out.nr));
        den.fetch_rho_spin_r(0, out.rho_a.data());
        den.fetch_rho_spin_r(1, out.rho_b.data());
    } else {
        out.rho_r.resize(static_cast<std::size_t>(out.nr));
        den.fetch_rho_r(out.rho_r.data());
    }
    out.energy_ry = static_cast<double>(last_energy_ha)
                    / cpu::pw::qe_units::rydberg_to_hartree;
    out.iterations = last_iter;
    out.converged = converged;
    result.energy = out.energy_ry;
    result.energy_ha = static_cast<double>(last_energy_ha);
    result.iterations = last_iter;
    result.converged = converged;
    out.result = result;
    return out;
}

/// Non-self-consistent free / total energy [Ry] with remapped density.
double nscf_energy_ry(
    const cpu::Frame& frame, const cpu::pw::internal::PWProblem& reference,
    double ecut_ry, int k_mesh, double mix_beta, int pulay_depth,
    cpu::xc::XCType xc, ATV::electronic::Smearing smearing, double degauss,
    int mp_order, int extra_bands, bool spin_polarized, int n_alpha, int n_beta,
    const std::vector<double>& rho_r0, const std::vector<double>& rho_a0,
    const std::vector<double>& rho_b0, double omega0_bohr) {
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem_fixed_basis(frame, reference, ecut_ry,
                                                    k_mesh, prob);
    // Every one of the 12 samples must differentiate the same finite
    // variational space, including all k-point Miller labels and FFT shapes.
    const auto same_labels = [](const auto& a, const auto& b) {
        return a.h == b.h && a.k == b.k && a.l == b.l && a.nx == b.nx
               && a.ny == b.ny && a.nz == b.nz;
    };
    bool fixed = same_labels(prob.gv0, reference.gv0)
                 && prob.gv_k.size() == reference.gv_k.size()
                 && prob.ws.fft_nx == reference.ws.fft_nx
                 && prob.ws.fft_ny == reference.ws.fft_ny
                 && prob.ws.fft_nz == reference.ws.fft_nz;
    for (std::size_t k = 0; fixed && k < prob.gv_k.size(); ++k)
        fixed = same_labels(prob.gv_k[k], reference.gv_k[k]);
    if (!fixed)
        throw std::logic_error("PW stress sample changed its reference basis");
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != ATV::electronic::Smearing::none);
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear) {
        // Match the reference SCF, including enough empty states in each
        // spin channel for its shared-chemical-potential filling problem.
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max({n_bands, n_occ_ceil + std::max(extra_bands, 0), 1});
    }
    const AccReal vol_bohr = prob.vol_bohr;
    const double scale = omega0_bohr / static_cast<double>(ws.omega);

    DeviceDensity den(prob, mix_beta, pulay_depth, xc, spin_polarized, n_alpha,
                      n_beta, n_elec);
    DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                            den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);

    if (spin_polarized) {
        den.load_rho_spin_r_scaled(rho_a0.data(), rho_b0.data(),
                                   static_cast<int>(rho_a0.size()), scale);
        den.hartree();
        den.exchange_correlation_spin();

        if (use_smear) {
            // A strain changes both spin spectra. Refill the complete set
            // with ONE chemical potential at fixed total electron count;
            // neither the reference occupations nor integer N_up/N_dn can
            // be reused when differentiating the Mermin free energy.
            const auto kind = to_smear_kind(smearing);
            const AccReal width =
                AccReal(degauss)
                * AccReal(cpu::pw::qe_units::rydberg_to_hartree);
            std::vector<std::vector<AccReal>> spectra(2 * n_kpts);
            std::vector<AccReal> all_evals, all_weights;
            for (int spin = 0; spin < 2; ++spin) {
                den.assemble_veff_spin(spin);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    auto& ev = spectra[spin * n_kpts + ik];
                    ev.assign(ws.eigenvalues.begin(), ws.eigenvalues.end());
                    ev.resize(static_cast<std::size_t>(n_bands));
                    for (const auto value : ev) {
                        all_evals.push_back(value);
                        all_weights.push_back(prob.kg.weights[ik]);
                    }
                }
            }
            const AccReal mu = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), width, kind, mp_order,
                AccReal(1));
            AccReal e_band = 0, entropy = 0;
            for (int spin = 0; spin < 2; ++spin)
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const auto& ev = spectra[spin * n_kpts + ik];
                    const AccReal weight = prob.kg.weights[ik];
                    const std::vector<AccReal> weights(ev.size(), weight);
                    const auto fill = ATV::electronic::fill_occupations(
                        ev, weights, mu, width, kind, mp_order, AccReal(1));
                    entropy += fill.entropy;
                    for (int band = 0; band < n_bands; ++band)
                        e_band += weight * ev[band] * fill.occ[band];
                }
            const auto red = den.reduce_spin(static_cast<double>(vol_bohr));
            // The frozen input density supplies the double-counting terms;
            // only the band occupations are reoptimized at this geometry.
            const AccReal e_one =
                e_band - AccReal(2) * AccReal(red.e_h) - AccReal(red.e_xc_int);
            const AccReal energy = e_one + AccReal(red.e_h) + AccReal(red.e_xc)
                                   + prob.ew_total - width * entropy;
            return static_cast<double>(energy)
                   / cpu::pw::qe_units::rydberg_to_hartree;
        }

        const int n_occ[2] = {n_alpha, n_beta};
        AccReal e_band = 0;
        for (int s = 0; s < 2; ++s) {
            if (n_occ[s] <= 0)
                continue;
            den.assemble_veff_spin(s);
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                const AccReal weight = prob.kg.weights[ik];
                for (int b = 0; b < n_occ[s]; ++b)
                    e_band += weight * ws.eigenvalues[b];
            }
        }

        const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
        auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            AccReal(s.e_xc), prob.ew_total);
        terms.e_one =
            e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
        terms.total =
            terms.e_one + AccReal(s.e_h) + AccReal(s.e_xc) + prob.ew_total;
        return static_cast<double>(terms.total)
               / cpu::pw::qe_units::rydberg_to_hartree;
    }

    den.load_rho_r_scaled(rho_r0.data(), static_cast<int>(rho_r0.size()),
                          scale);
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    AccReal e_band = 0;
    AccReal entropy_sum = AccReal(0);

    if (use_smear) {
        const ATV::electronic::Smearing smear_kind = to_smear_kind(smearing);
        const AccReal degauss_ha =
            AccReal(degauss) * AccReal(cpu::pw::qe_units::rydberg_to_hartree);

        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
        for (int ik = 0; ik < n_kpts; ++ik) {
            solver.bind_kpoint(ik);
            solver.solve(ws, 40);
            evals_k[static_cast<std::size_t>(ik)].assign(ws.eigenvalues.begin(),
                                                         ws.eigenvalues.end());
            if (static_cast<int>(evals_k[static_cast<std::size_t>(ik)].size())
                > n_bands)
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));
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

        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            std::vector<AccReal> w_one(ev.size(), w);
            auto fill = ATV::electronic::fill_occupations(
                ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
            entropy_sum += fill.entropy;
            for (std::size_t b = 0; b < ev.size(); ++b) {
                const AccReal f_b = ATV::electronic::smear_f_band(
                    ev[b], mu_ha, degauss_ha, smear_kind, mp_order);
                e_band += w * ev[b] * f_b;
            }
        }

        const auto s = den.reduce(static_cast<double>(vol_bohr));
        const auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            AccReal(s.e_xc), prob.ew_total);
        const AccReal e_ha = terms.total + (-degauss_ha * entropy_sum);
        return static_cast<double>(e_ha)
               / cpu::pw::qe_units::rydberg_to_hartree;
    }

    for (int ik = 0; ik < n_kpts; ++ik) {
        solver.bind_kpoint(ik);
        solver.solve(ws, 40);
        const AccReal weight = prob.kg.weights[ik];
        for (int b = 0; b < n_bands; ++b)
            e_band += weight * ws.eigenvalues[b] * AccReal(2);
    }

    const auto s = den.reduce(static_cast<double>(vol_bohr));
    const auto terms = cpu::pw::internal::combine_energy_terms(
        e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
        AccReal(s.e_xc), prob.ew_total);
    return static_cast<double>(terms.total)
           / cpu::pw::qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6> pw_stress_analytic_impl(
    const cpu::Frame& frame,
    const ::ATV::electronic::NormConservingPotential& potential, double ecut_ry,
    int k_mesh, double tol, int max_iter, double mix_beta, int pulay_depth,
    cpu::xc::XCType xc, ATV::electronic::Smearing smearing, double degauss,
    int mp_order, int extra_bands, bool spin_polarized, int multiplicity,
    double* energy_ry_out, cpu::pw::PWResult* result_out) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("cuda::pw::pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: non-positive cell volume");
    if (smearing != ATV::electronic::Smearing::none && !(degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::pw_stress: degauss must be > 0 [Ry] when smearing is "
            "enabled");
    if (spin_polarized && multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: multiplicity must be >= 1 when "
            "spin_polarized");

    const RefScf ref =
        run_device_scf(frame, potential, ecut_ry, k_mesh, tol, max_iter,
                       mix_beta, pulay_depth, xc, smearing, degauss, mp_order,
                       extra_bands, spin_polarized, multiplicity);
    if (energy_ry_out)
        *energy_ry_out = ref.energy_ry;
    if (result_out)
        *result_out = ref.result;
    if (!ref.converged) {
        // No core/log.h from .cu — printf matches forces-scf breadcrumb.
        std::printf("[pw] cuda pw_stress: SCF not converged after %d iters; "
                    "using last density for HF stress (E=%.8f Ry)\n",
                    ref.iterations, ref.energy_ry);
    } else {
        std::printf(
            "[pw] cuda stress-scf done iter=%d energy_ry=%.8f%s%s\n",
            ref.iterations, ref.energy_ry, spin_polarized ? " spin=1" : "",
            smearing != ATV::electronic::Smearing::none ? " smear=1" : "");
    }

    // Hellmann–Feynman strain-FD of the NSCF free/total energy (one SCF).
    // Same step as CPU `pw_stress_analytic_impl`.
    constexpr double h = 2e-3;
    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = cpu::pw::qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }
        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);

        const double ep_ry =
            nscf_energy_ry(fp, *ref.problem, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, smearing, degauss, mp_order,
                           extra_bands, spin_polarized, ref.n_alpha, ref.n_beta,
                           ref.rho_r, ref.rho_a, ref.rho_b, ref.omega_bohr);
        const double em_ry =
            nscf_energy_ry(fm, *ref.problem, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, smearing, degauss, mp_order,
                           extra_bands, spin_polarized, ref.n_alpha, ref.n_beta,
                           ref.rho_r, ref.rho_a, ref.rho_b, ref.omega_bohr);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (E = F under smear).
        // A symmetric off-diagonal probe varies epsilon_ab AND epsilon_ba.
        // Divide that derivative by two to publish the physical tensor entry.
        stress[static_cast<std::size_t>(v)] =
            -dE_dev / (vol * (a == b ? 1.0 : 2.0));
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
