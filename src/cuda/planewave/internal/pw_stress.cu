/// @file cuda/planewave/internal/pw_stress.cu
/// @brief Device frozen-density cell stress (one SCF + NSCF strain-FD).
///
/// Restricted/collinear LSDA or PBE, with fixed or fractional occupations.
/// Each strained spin-smearing spectrum is refilled at a shared chemical
/// potential and includes the same free-energy correction as its SCF.

#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

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

/// Non-self-consistent free / total energy [Ry] with remapped density.
double nscf_energy_ry(
    const cpu::Frame& frame, cpu::pw::internal::PWStrainReference reference,
    double ecut_ry, int k_mesh, double mix_beta, int pulay_depth,
    cpu::xc::XCType xc, ATV::electronic::Smearing smearing, double degauss,
    int mp_order, int extra_bands, bool spin_polarized, int n_alpha, int n_beta,
    const std::vector<double>& rho_r0, const std::vector<double>& rho_a0,
    const std::vector<double>& rho_b0, double omega0_bohr,
    cudaStream_t stream) {
    cpu::pw::internal::PWSetup prob;
    cpu::pw::internal::PWWorkspace ws;
    cpu::pw::internal::setup_pw_problem_fixed_basis(frame, reference, ecut_ry,
                                                    k_mesh, prob, ws);
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_elec = prob.ions.n_elec;
    const bool use_smear = (smearing != ATV::electronic::Smearing::none);
    const int n_kpts = static_cast<int>(prob.mesh.kg.kpoints.size());
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear) {
        // Match the reference SCF, including enough empty states in each
        // spin channel for its shared-chemical-potential filling problem.
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max({n_bands, n_occ_ceil + std::max(extra_bands, 0), 1});
    }
    const AccReal vol_bohr = prob.ions.vol_bohr;
    const double scale = omega0_bohr / static_cast<double>(ws.omega);

    DeviceDensity den(prob.grid(), ws.density().total, mix_beta, pulay_depth,
                      xc, spin_polarized, n_alpha, n_beta, n_elec, stream);
    DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                            den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob.mesh, prob.ions.positions, prob.ions.vol_bohr,
                         ws.wfc);

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
                    solver.solve(ws.eigenvalues, 40);
                    auto& ev = spectra[spin * n_kpts + ik];
                    ev.assign(ws.eigenvalues.begin(), ws.eigenvalues.end());
                    ev.resize(static_cast<std::size_t>(n_bands));
                    for (const auto value : ev) {
                        all_evals.push_back(value);
                        all_weights.push_back(prob.mesh.kg.weights[ik]);
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
                    const AccReal weight = prob.mesh.kg.weights[ik];
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
                                   + prob.ions.ew_total - width * entropy;
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
                solver.solve(ws.eigenvalues, 40);
                const AccReal weight = prob.mesh.kg.weights[ik];
                for (int b = 0; b < n_occ[s]; ++b)
                    e_band += weight * ws.eigenvalues[b];
            }
        }

        const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
        auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            AccReal(s.e_xc), prob.ions.ew_total);
        terms.e_one =
            e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
        terms.total =
            terms.e_one + AccReal(s.e_h) + AccReal(s.e_xc) + prob.ions.ew_total;
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
            solver.solve(ws.eigenvalues, 40);
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
            const AccReal w =
                prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
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
            const AccReal w =
                prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
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
            AccReal(s.e_xc), prob.ions.ew_total);
        const AccReal e_ha = terms.total + (-degauss_ha * entropy_sum);
        return static_cast<double>(e_ha)
               / cpu::pw::qe_units::rydberg_to_hartree;
    }

    for (int ik = 0; ik < n_kpts; ++ik) {
        solver.bind_kpoint(ik);
        solver.solve(ws.eigenvalues, 40);
        const AccReal weight = prob.mesh.kg.weights[ik];
        for (int b = 0; b < n_bands; ++b)
            e_band += weight * ws.eigenvalues[b] * AccReal(2);
    }

    const auto s = den.reduce(static_cast<double>(vol_bohr));
    const auto terms = cpu::pw::internal::combine_energy_terms(
        e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
        AccReal(s.e_xc), prob.ions.ew_total);
    return static_cast<double>(terms.total)
           / cpu::pw::qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6>
pw_stress_from_state(const cpu::Frame& frame, const PW& method,
                     cpu::pw::internal::PWStrainReference reference,
                     DeviceDensity& density, int n_alpha, int n_beta) {
    const bool spin_polarized = method.spin_polarized;
    const auto stream = density.stream();
    const int nr = density.nr();
    std::vector<double> rho, rho_a, rho_b;
    if (spin_polarized) {
        rho_a.resize(nr);
        rho_b.resize(nr);
        density.fetch_rho_spin_r(0, rho_a.data());
        density.fetch_rho_spin_r(1, rho_b.data());
    } else {
        rho.resize(nr);
        density.fetch_rho_r(rho.data());
    }
    const double omega0 = density.omega();
    const double ecut_ry = method.ecut_ry;
    const int k_mesh = method.k_mesh;
    const double mix_beta = method.mix_beta;
    const int pulay_depth = method.pulay_depth;
    const auto xc = method.xc;
    const auto smearing = method.smearing;
    const double degauss = method.degauss;
    const int mp_order = method.mp_order;
    const int extra_bands = method.extra_bands;

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

        const double ep_ry = nscf_energy_ry(
            fp, reference, ecut_ry, k_mesh, mix_beta, pulay_depth, xc, smearing,
            degauss, mp_order, extra_bands, spin_polarized, n_alpha, n_beta,
            rho, rho_a, rho_b, omega0, stream);
        const double em_ry = nscf_energy_ry(
            fm, reference, ecut_ry, k_mesh, mix_beta, pulay_depth, xc, smearing,
            degauss, mp_order, extra_bands, spin_polarized, n_alpha, n_beta,
            rho, rho_a, rho_b, omega0, stream);
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
