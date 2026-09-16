/// @file cpu/scf/cisd.cpp
/// @brief CIS(D) doubles correction (Head-Gordon 1994) on closed-shell CIS.
///
/// Spin-orbital formulas, eqs. (8), (13), (14) of
/// Head-Gordon, Rico, Oumi & Lee, Chem. Phys. Lett. 219, 21 (1994),
/// DOI 10.1016/0009-2614(94)00070-0. Spatial TDA amplitudes from
/// `TDHFPack::tda` (unit-norm Σ X² = 1) map to spin-orbitals as
/// b_{iα}^{aα} = b_{iβ}^{aβ} = X_ia / √2.

#include "atomiverse/cpu/scf/cisd.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"
#include "atomiverse/cpu/scf/internal/tdhf_pack.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

/// CIS(D) correction U + V [Hartree] for one CIS root (spin-orbital).
double cisd_correction_ha(const xt::xtensor<double, 4>& eri_mo,
                          const RealVector& eps, int n_occ_sp, int n_mo,
                          const RealVector& X_spatial, // length n_occ*n_virt
                          double omega_ha) {
    const int n_vir_sp = n_mo - n_occ_sp;
    // Spin-orbitals: α-occ, β-occ, α-vir, β-vir
    std::vector<SpinOrbital> occ, vir;
    occ.reserve(static_cast<std::size_t>(2 * n_occ_sp));
    vir.reserve(static_cast<std::size_t>(2 * n_vir_sp));
    for (int i = 0; i < n_occ_sp; ++i)
        occ.push_back(SpinOrbital{i, 0});
    for (int i = 0; i < n_occ_sp; ++i)
        occ.push_back(SpinOrbital{i, 1});
    for (int a = 0; a < n_vir_sp; ++a)
        vir.push_back(SpinOrbital{n_occ_sp + a, 0});
    for (int a = 0; a < n_vir_sp; ++a)
        vir.push_back(SpinOrbital{n_occ_sp + a, 1});
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(vir.size());

    // Spatial CIS X (unit-norm) → spin-orbital b: b_α = b_β = X/√2
    // Composite spatial index matches OrbitalHessian: a·n_occ + i
    std::vector<double> b(static_cast<std::size_t>(no * nv), 0.0);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < n_occ_sp; ++i) {
        for (int a = 0; a < n_vir_sp; ++a) {
            const double x =
                X_spatial(static_cast<std::size_t>(a * n_occ_sp + i));
            const double bs = x * inv_sqrt2;
            // α block: occ index i, vir index a
            b[static_cast<std::size_t>(i * nv + a)] = bs;
            // β block: occ index n_occ_sp+i, vir index n_vir_sp+a
            b[static_cast<std::size_t>((n_occ_sp + i) * nv + (n_vir_sp + a))] =
                bs;
        }
    }

    auto b_at = [&](int ii, int aa) -> double {
        return b[static_cast<std::size_t>(ii * nv + aa)];
    };
    auto eps_so = [&](const SpinOrbital& o) -> double {
        return eps(static_cast<std::size_t>(o.spat));
    };

    // MP2 doubles a_ij^ab = −⟨ab||ij⟩ / (ε_a + ε_b − ε_i − ε_j)
    // Stored [no, no, nv, nv]
    xt::xtensor<double, 4> a_amp = xt::zeros<double>(
        {static_cast<std::size_t>(no), static_cast<std::size_t>(no),
         static_cast<std::size_t>(nv), static_cast<std::size_t>(nv)});
    for (int ii = 0; ii < no; ++ii) {
        for (int jj = 0; jj < no; ++jj) {
            for (int aa = 0; aa < nv; ++aa) {
                for (int bb = 0; bb < nv; ++bb) {
                    const double denom =
                        eps_so(vir[static_cast<std::size_t>(aa)])
                        + eps_so(vir[static_cast<std::size_t>(bb)])
                        - eps_so(occ[static_cast<std::size_t>(ii)])
                        - eps_so(occ[static_cast<std::size_t>(jj)]);
                    if (std::abs(denom) < 1e-12)
                        throw std::runtime_error(
                            "RHF::cisd: near-zero MP2 doubles denominator "
                            "(degenerate or non-aufbau reference)");
                    const double abij =
                        antisym_eri(eri_mo, vir[static_cast<std::size_t>(aa)],
                                    vir[static_cast<std::size_t>(bb)],
                                    occ[static_cast<std::size_t>(ii)],
                                    occ[static_cast<std::size_t>(jj)]);
                    a_amp(ii, jj, aa, bb) = -abij / denom;
                }
            }
        }
    }

    // ũ_ij^ab (eq. 8) and V = −¼ Σ (ũ)² / (Δ − ω)
    double V = 0.0;
    xt::xtensor<double, 4> u_tilde = xt::zeros<double>(
        {static_cast<std::size_t>(no), static_cast<std::size_t>(no),
         static_cast<std::size_t>(nv), static_cast<std::size_t>(nv)});
    for (int ii = 0; ii < no; ++ii) {
        for (int jj = 0; jj < no; ++jj) {
            for (int aa = 0; aa < nv; ++aa) {
                for (int bb = 0; bb < nv; ++bb) {
                    double s = 0.0;
                    for (int cc = 0; cc < nv; ++cc) {
                        // (ab||cj) b_i^c − (ab||ci) b_j^c
                        s += antisym_eri(eri_mo,
                                         vir[static_cast<std::size_t>(aa)],
                                         vir[static_cast<std::size_t>(bb)],
                                         vir[static_cast<std::size_t>(cc)],
                                         occ[static_cast<std::size_t>(jj)])
                             * b_at(ii, cc);
                        s -= antisym_eri(eri_mo,
                                         vir[static_cast<std::size_t>(aa)],
                                         vir[static_cast<std::size_t>(bb)],
                                         vir[static_cast<std::size_t>(cc)],
                                         occ[static_cast<std::size_t>(ii)])
                             * b_at(jj, cc);
                    }
                    for (int kk = 0; kk < no; ++kk) {
                        // (ka||ij) b_k^b − (kb||ij) b_k^a
                        s += antisym_eri(eri_mo,
                                         occ[static_cast<std::size_t>(kk)],
                                         vir[static_cast<std::size_t>(aa)],
                                         occ[static_cast<std::size_t>(ii)],
                                         occ[static_cast<std::size_t>(jj)])
                             * b_at(kk, bb);
                        s -= antisym_eri(eri_mo,
                                         occ[static_cast<std::size_t>(kk)],
                                         vir[static_cast<std::size_t>(bb)],
                                         occ[static_cast<std::size_t>(ii)],
                                         occ[static_cast<std::size_t>(jj)])
                             * b_at(kk, aa);
                    }
                    u_tilde(ii, jj, aa, bb) = s;
                    const double denom =
                        eps_so(vir[static_cast<std::size_t>(aa)])
                        + eps_so(vir[static_cast<std::size_t>(bb)])
                        - eps_so(occ[static_cast<std::size_t>(ii)])
                        - eps_so(occ[static_cast<std::size_t>(jj)]) - omega_ha;
                    if (std::abs(denom) < 1e-12)
                        throw std::runtime_error(
                            "RHF::cisd: near-zero CIS(D) doubles denominator "
                            "(Δ − ω)");
                    V += s * s / denom;
                }
            }
        }
    }
    V *= -0.25;

    // v_i^a (eq. 13) and U = Σ b_i^a v_i^a
    // v_i^a = ½ Σ_jkbc ⟨jk||bc⟩ (b_i^b a_jk^{ca} + b_j^a a_ik^{cb} + 2 b_j^b a_ik^{ac})
    double U = 0.0;
    for (int ii = 0; ii < no; ++ii) {
        for (int aa = 0; aa < nv; ++aa) {
            double v_ia = 0.0;
            for (int jj = 0; jj < no; ++jj) {
                for (int kk = 0; kk < no; ++kk) {
                    for (int bb = 0; bb < nv; ++bb) {
                        for (int cc = 0; cc < nv; ++cc) {
                            const double jklbc = antisym_eri(
                                eri_mo, occ[static_cast<std::size_t>(jj)],
                                occ[static_cast<std::size_t>(kk)],
                                vir[static_cast<std::size_t>(bb)],
                                vir[static_cast<std::size_t>(cc)]);
                            const double term =
                                b_at(ii, bb) * a_amp(jj, kk, cc, aa)
                                + b_at(jj, aa) * a_amp(ii, kk, cc, bb)
                                + 2.0 * b_at(jj, bb) * a_amp(ii, kk, aa, cc);
                            v_ia += jklbc * term;
                        }
                    }
                }
            }
            v_ia *= 0.5;
            U += b_at(ii, aa) * v_ia;
        }
    }
    return U + V;
}

} // namespace

CISDResult RHF::cisd(int n_states) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::cisd: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    // Parent CIS roots (singlet TDA) — unit-norm spatial X.
    const internal::OrbitalHessian hessian{det, eri_ao};
    const internal::TDHFPack tdhf_pack{hessian, internal::Channel::singlet};
    const internal::TDHFPack::Roots roots = tdhf_pack.tda(n_states);

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p)
        if (f(static_cast<std::size_t>(p)) > 0.0)
            ++n_occ;
    if (n_occ == 0 || n_occ >= n_mo)
        throw std::invalid_argument(
            "RHF::cisd: no virtual orbitals (e.g. He/STO-3G — no fallback)");

    // Full MO ERI (pq|rs) chemist notation.
    const xt::xtensor<double, 4> eri_mo = ao2mo_full(eri_ao, C, n_mo);

    CISDResult out;
    out.energy_ev.resize(static_cast<std::size_t>(n_states));
    out.cis_energy_ev.resize(static_cast<std::size_t>(n_states));
    out.doubles_correction_ev.resize(static_cast<std::size_t>(n_states));

    for (int s = 0; s < n_states; ++s) {
        const double omega = roots.omega(static_cast<std::size_t>(s));
        RealVector X = xt::zeros<double>({roots.X.shape()[1]});
        for (std::size_t p = 0; p < X.size(); ++p)
            X(p) = roots.X(static_cast<std::size_t>(s), p);

        const double corr =
            cisd_correction_ha(eri_mo, eps, n_occ, n_mo, X, omega);
        out.cis_energy_ev[static_cast<std::size_t>(s)] =
            omega * units::kHartreeInEv;
        out.doubles_correction_ev[static_cast<std::size_t>(s)] =
            corr * units::kHartreeInEv;
        out.energy_ev[static_cast<std::size_t>(s)] =
            (omega + corr) * units::kHartreeInEv;
    }
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
