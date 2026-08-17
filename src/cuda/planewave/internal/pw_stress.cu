/// @file cuda/planewave/internal/pw_stress.cu
/// @brief Device Nielsen–Martin analytic cell stress (one SCF + NSCF strain-FD).
///
/// Restricted fixed closed shell, restricted metallic smear (Mermin free
/// energy), and collinear fixed-occupation LSDA spin (ρ_↑/ρ_↓ remap).

#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"

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

cpu::pw::internal::SmearKind to_smear_kind(cpu::pw::Smearing s) {
    using cpu::pw::Smearing;
    using cpu::pw::internal::SmearKind;
    if (s == Smearing::methfessel_paxton)
        return SmearKind::methfessel_paxton;
    if (s == Smearing::gaussian)
        return SmearKind::gaussian;
    return SmearKind::fermi_dirac;
}

/// Device SCF at `frame`; fills free/total energy and returns host ρ(r) + Ω.
/// Collinear spin: rho_r is unused; rho_a / rho_b hold the spin channels.
struct RefScf {
    std::vector<double> rho_r;
    std::vector<double> rho_a;
    std::vector<double> rho_b;
    double omega_bohr = 0;
    double energy_ry = 0;
    int nr = 0;
    int iterations = 0;
    bool converged = false;
    bool spin = false;
    int n_alpha = 0;
    int n_beta = 0;
};

RefScf run_device_scf(const cpu::Frame& frame, const std::string& upf_path,
                      const std::string& gth_name, double ecut_ry, int k_mesh,
                      double tol, int max_iter, double mix_beta,
                      int pulay_depth, cpu::scf::XCType xc,
                      cpu::pw::Smearing smearing, double degauss, int mp_order,
                      int extra_bands, bool spin_polarized, int multiplicity) {
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry,
                                        k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != cpu::pw::Smearing::none);

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

    const cpu::pw::internal::SmearKind smear_kind = to_smear_kind(smearing);
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
            const AccReal mu_ha = cpu::pw::internal::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order, AccReal(1));
            AccReal e_band = 0;
            AccReal entropy_sum = 0;
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
                    auto fill = cpu::pw::internal::fill_occupations(
                        ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order,
                        AccReal(1));
                    entropy_sum += fill.entropy;
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
                AccReal(red.e_loc), ws.E_xc, prob.ew_total, ws.E_aug);
            terms.e_one = e_band - AccReal(2.0) * AccReal(red.e_h)
                          - AccReal(red.e_xc_int);
            last_energy_ha = terms.e_one + AccReal(red.e_h) + ws.E_xc
                             + prob.ew_total + ws.E_aug
                             - degauss_ha * entropy_sum;
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
                ws.E_xc, prob.ew_total, ws.E_aug);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            terms.total = terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total
                          + ws.E_aug;
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
            const AccReal mu_ha = cpu::pw::internal::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order);

            AccReal entropy_sum = AccReal(0);
            std::vector<std::vector<AccReal>> occ_k(
                static_cast<std::size_t>(n_kpts));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                std::vector<AccReal> w_one(ev.size(), w);
                auto fill = cpu::pw::internal::fill_occupations(
                    ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
                occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
                for (std::size_t b = 0; b < ev.size(); ++b)
                    occ_k[static_cast<std::size_t>(ik)][b] =
                        cpu::pw::internal::smear_f_band(
                            ev[b], mu_ha, degauss_ha, smear_kind, mp_order);
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
                ws.E_xc, prob.ew_total, ws.E_aug);
            // Mermin free energy: F = E_KS − σ · S  (σ = degauss in Ha).
            last_energy_ha = terms.total + (-degauss_ha * entropy_sum);
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
                ws.E_xc, prob.ew_total, ws.E_aug);
            last_energy_ha = terms.total;
            last_iter = iter + 1;
            if (conv.step(AccReal(s.resid))) {
                converged = true;
                break;
            }
        }
    }

    RefScf out;
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
    return out;
}

/// Non-self-consistent free / total energy [Ry] with remapped density.
double nscf_energy_ry(const cpu::Frame& frame, const std::string& upf_path,
                      const std::string& gth_name, double ecut_ry, int k_mesh,
                      double mix_beta, int pulay_depth, cpu::scf::XCType xc,
                      cpu::pw::Smearing smearing, double degauss, int mp_order,
                      int extra_bands, bool spin_polarized, int n_alpha,
                      int n_beta, const std::vector<double>& rho_r0,
                      const std::vector<double>& rho_a0,
                      const std::vector<double>& rho_b0, double omega0_bohr) {
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry,
                                        k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != cpu::pw::Smearing::none);
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear && !spin_polarized) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = n_occ_ceil + std::max(extra_bands, 0);
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
            AccReal(s.e_xc), prob.ew_total, ws.E_aug);
        terms.e_one =
            e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
        terms.total = terms.e_one + AccReal(s.e_h) + AccReal(s.e_xc)
                      + prob.ew_total + ws.E_aug;
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
        const cpu::pw::internal::SmearKind smear_kind = to_smear_kind(smearing);
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
        const AccReal mu_ha = cpu::pw::internal::find_fermi_level(
            all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
            mp_order);

        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            std::vector<AccReal> w_one(ev.size(), w);
            auto fill = cpu::pw::internal::fill_occupations(
                ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
            entropy_sum += fill.entropy;
            for (std::size_t b = 0; b < ev.size(); ++b) {
                const AccReal f_b = cpu::pw::internal::smear_f_band(
                    ev[b], mu_ha, degauss_ha, smear_kind, mp_order);
                e_band += w * ev[b] * f_b;
            }
        }

        const auto s = den.reduce(static_cast<double>(vol_bohr));
        const auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            AccReal(s.e_xc), prob.ew_total, ws.E_aug);
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
        AccReal(s.e_xc), prob.ew_total, ws.E_aug);
    return static_cast<double>(terms.total)
           / cpu::pw::qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6> pw_stress_analytic_impl(
    const cpu::Frame& frame, const std::string& upf_path,
    const std::string& gth_name, double ecut_ry, int k_mesh, double tol,
    int max_iter, double mix_beta, int pulay_depth, cpu::scf::XCType xc,
    cpu::pw::Smearing smearing, double degauss, int mp_order, int extra_bands,
    bool spin_polarized, int multiplicity, double* energy_ry_out) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("cuda::pw::pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: non-positive cell volume");
    if (smearing != cpu::pw::Smearing::none && !(degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::pw_stress: degauss must be > 0 [Ry] when smearing is "
            "enabled");
    if (spin_polarized && multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: multiplicity must be >= 1 when "
            "spin_polarized");

    const RefScf ref =
        run_device_scf(frame, upf_path, gth_name, ecut_ry, k_mesh, tol,
                       max_iter, mix_beta, pulay_depth, xc, smearing, degauss,
                       mp_order, extra_bands, spin_polarized, multiplicity);
    if (energy_ry_out)
        *energy_ry_out = ref.energy_ry;
    if (!ref.converged) {
        // No core/log.h from .cu — printf matches forces-scf breadcrumb.
        std::printf("[pw] cuda pw_stress: SCF not converged after %d iters; "
                    "using last density for HF stress (E=%.8f Ry)\n",
                    ref.iterations, ref.energy_ry);
    } else {
        std::printf("[pw] cuda stress-scf done iter=%d energy_ry=%.8f%s%s\n",
                    ref.iterations, ref.energy_ry,
                    spin_polarized ? " spin=1" : "",
                    smearing != cpu::pw::Smearing::none ? " smear=1" : "");
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
            nscf_energy_ry(fp, upf_path, gth_name, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, smearing, degauss, mp_order,
                           extra_bands, spin_polarized, ref.n_alpha, ref.n_beta,
                           ref.rho_r, ref.rho_a, ref.rho_b, ref.omega_bohr);
        const double em_ry =
            nscf_energy_ry(fm, upf_path, gth_name, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, smearing, degauss, mp_order,
                           extra_bands, spin_polarized, ref.n_alpha, ref.n_beta,
                           ref.rho_r, ref.rho_a, ref.rho_b, ref.omega_bohr);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (E = F under smear).
        stress[static_cast<std::size_t>(v)] = -dE_dev / vol;
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
