/// @file src/cpu/semiempirical/dftb3.cpp
/// SCC-DFTB3 multi-element (H/C/N/O) with transcribed mio-1-1 SK tables.
/// All ordered pairs share `MioSkTable` / `MioSketchPair` with DFTB0/DFTB2, so
/// DFTB3 reduces to DFTB2 exactly when charge fluctuations vanish. Nuclear
/// forces are the CLOSED-FORM gradient at the converged SCC density —
/// Hellmann–Feynman + Pulay + the second- AND third-order charge-kernel
/// radial derivatives + repulsive.
#include "atomiverse/cpu/semiempirical/dftb3.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_sk_tables.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/internal/dftb3_scc.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_gamma.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_block.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_table.h"
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::DftbElementRow;
using basis_data::DftbPairRepRow;
using basis_data::DftbPairSkRow;
using basis_data::kDftbElements;
using basis_data::kDftbNElements;
using basis_data::kDftbNPairRep;
using basis_data::kDftbNPairSk;
using basis_data::kDftbPairRep;
using basis_data::kDftbPairSk;

const DftbElementRow& element_row(int z) {
    for (std::size_t i = 0; i < kDftbNElements; ++i)
        if (kDftbElements[i].z == z)
            return kDftbElements[i];
    throw std::runtime_error("DFTB3: unsupported element Z=" + std::to_string(z)
                             + " (static mio-style tables cover H/C/N/O only)");
}

const DftbPairSkRow& pair_sk(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairSk; ++i)
        if (kDftbPairSk[i].z_a == za && kDftbPairSk[i].z_b == zb)
            return kDftbPairSk[i];
    throw std::runtime_error("DFTB3: no SK table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

const DftbPairRepRow& pair_rep(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairRep; ++i)
        if (kDftbPairRep[i].z_a == za && kDftbPairRep[i].z_b == zb)
            return kDftbPairRep[i];
    throw std::runtime_error("DFTB3: no repulsive table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

/// mio-1-1 pair cutoff [Å] (SK grid + distFudge).
double pair_rcut_A(const DftbElementRow& ea, const DftbElementRow& eb) {
    return internal::MioSkTable{ea.z, eb.z}.cutoff();
}

/// Pair repulsive energy [eV] from the transcribed mio-1-1 spline.
double pair_repulsive_eV(int za, int zb, double r_A) {
    return internal::MioSkTable{za, zb}.repulsive(r_A);
}

/// Radial derivative of @ref pair_repulsive_eV [eV/Å].
double pair_repulsive_dE_dr(int za, int zb, double r_A) {
    return internal::MioSkTable{za, zb}.d_repulsive_dr(r_A);
}

// ── Γ-point lattice walk (same shape as DFTB2) ─────────────────────────

struct LatVec {
    double tx = 0, ty = 0, tz = 0;
};

std::vector<LatVec> enumerate_lattice(const Mat3& cell, double max_cut) {
    double col_len[3];
    for (int d = 0; d < 3; ++d) {
        const double cx = static_cast<double>(cell(0, d));
        const double cy = static_cast<double>(cell(1, d));
        const double cz = static_cast<double>(cell(2, d));
        col_len[d] = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (col_len[d] < 1e-9)
            throw std::runtime_error("DFTB3: degenerate cell edge");
    }
    int nmax[3];
    for (int d = 0; d < 3; ++d)
        nmax[d] =
            std::max(1, static_cast<int>(std::ceil(max_cut / col_len[d])) + 1);
    std::vector<LatVec> out;
    for (int n0 = -nmax[0]; n0 <= nmax[0]; ++n0)
        for (int n1 = -nmax[1]; n1 <= nmax[1]; ++n1)
            for (int n2 = -nmax[2]; n2 <= nmax[2]; ++n2) {
                const double tx = n0 * static_cast<double>(cell(0, 0))
                                  + n1 * static_cast<double>(cell(0, 1))
                                  + n2 * static_cast<double>(cell(0, 2));
                const double ty = n0 * static_cast<double>(cell(1, 0))
                                  + n1 * static_cast<double>(cell(1, 1))
                                  + n2 * static_cast<double>(cell(1, 2));
                const double tz = n0 * static_cast<double>(cell(2, 0))
                                  + n1 * static_cast<double>(cell(2, 1))
                                  + n2 * static_cast<double>(cell(2, 2));
                out.push_back({tx, ty, tz});
            }
    return out;
}

std::vector<LatVec>
lattice_images_for(const cpu::Box& box,
                   const std::vector<const DftbElementRow*>& el) {
    double max_cut = 0;
    for (const auto* e : el)
        max_cut = std::max(max_cut, e->r_cut);
    for (std::size_t i = 0; i < basis_data::kMioSkNPairs; ++i) {
        const auto& p = basis_data::kMioSkPairs[i];
        max_cut = std::max(max_cut, internal::MioSkTable{p.za, p.zb}.cutoff());
    }
    if (max_cut < 1e-12)
        max_cut = 1.0;
    return enumerate_lattice(box.cell(), max_cut);
}

/// DFTB+ Elstner γ(R) [eV] at one separation (not lattice-summed).
double gamma1(double r, double ua, double ub) {
    return internal::dftb_gamma::gamma_eV(r, ua, ub);
}

double dgamma1_dr(double r, double ua, double ub) {
    return internal::dftb_gamma::dgamma_dr_eV(r, ua, ub);
}

/// Γ_AB = ∂γ/∂q_A at one r (not lattice-summed).
double gamma_charge1(double r, double ua, double ub, double uda) {
    return internal::dftb_gamma::gamma_charge_eV(r, ua, ub, uda);
}

double gamma_charge1_dr(double r, double ua, double ub, double uda) {
    return internal::dftb_gamma::gamma_charge_dr_eV(r, ua, ub, uda);
}

/// Lattice-summed γ_AB [eV] (DFTB2-style, cut at SK rcut).
double gamma_pair_lat(int a, int b,
                      const std::vector<const DftbElementRow*>& el,
                      const StorageReal* x, const StorageReal* y,
                      const StorageReal* z, const cpu::Box& box,
                      const std::vector<LatVec>& images) {
    const double ua = el[static_cast<std::size_t>(a)]->hubbard_u;
    const double ub = el[static_cast<std::size_t>(b)]->hubbard_u;
    const double rcut = pair_rcut_A(*el[static_cast<std::size_t>(a)],
                                    *el[static_cast<std::size_t>(b)]);
    const double xa = static_cast<double>(x[a]);
    const double ya = static_cast<double>(y[a]);
    const double za = static_cast<double>(z[a]);
    const double xb = static_cast<double>(x[b]);
    const double yb = static_cast<double>(y[b]);
    const double zb = static_cast<double>(z[b]);
    double g = 0;
    for (const auto& T : images) {
        const double dx = xb + T.tx - xa;
        const double dy = yb + T.ty - ya;
        const double dz = zb + T.tz - za;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-12) {
            if (a == b)
                g += ua;
            continue;
        }
        if (r > rcut)
            continue;
        g += gamma1(r, ua, ub);
    }
    return g;
}

double gamma_charge_pair_lat(int a, int b,
                             const std::vector<const DftbElementRow*>& el,
                             const std::vector<double>& hubbard_du_ev,
                             const StorageReal* x, const StorageReal* y,
                             const StorageReal* z, const cpu::Box& box,
                             const std::vector<LatVec>& images) {
    // Γ_AB = ∂γ_AB/∂q_A lattice sum.
    const double ua = el[static_cast<std::size_t>(a)]->hubbard_u;
    const double ub = el[static_cast<std::size_t>(b)]->hubbard_u;
    const double uda = hubbard_du_ev[static_cast<std::size_t>(a)];
    const double rcut = pair_rcut_A(*el[static_cast<std::size_t>(a)],
                                    *el[static_cast<std::size_t>(b)]);
    const double xa = static_cast<double>(x[a]);
    const double ya = static_cast<double>(y[a]);
    const double za = static_cast<double>(z[a]);
    const double xb = static_cast<double>(x[b]);
    const double yb = static_cast<double>(y[b]);
    const double zb = static_cast<double>(z[b]);
    double g = 0;
    for (const auto& T : images) {
        const double dx = xb + T.tx - xa;
        const double dy = yb + T.ty - ya;
        const double dz = zb + T.tz - za;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-12) {
            if (a == b)
                g += uda;
            continue;
        }
        if (r > rcut)
            continue;
        g += gamma_charge1(r, ua, ub, uda);
    }
    return g;
}

/// One converged DFTB3 solve at a frozen geometry.
struct SccSolution {
    double e_h0 = 0;  ///< Tr(ρH⁰) [eV]
    double e2 = 0;    ///< E^(2) [eV]
    double e3 = 0;    ///< E^(3) [eV]
    double e_rep = 0; ///< pair repulsive [eV]
    double residual = 0;
    int iterations = 0;
    std::vector<double> dq;
    /// Converged MO coefficients and eigenvalues [eV] of the SCC Hamiltonian —
    /// the analytic gradient contracts exactly the solve that converged.
    linalg::RealMatrix C;
    linalg::RealVector eps;
    int n_occ = 0;
    /// The charge kernel of this geometry: γ, Γ and their radial derivatives.
    std::optional<internal::Dftb3Scc> scc;

    /// @brief Total electronic + repulsive energy [eV].
    double energy() const { return e_h0 + e2 + e3 + e_rep; }
};

/// The one DFTB3 self-consistency loop.
///
/// Charges are Anderson-mixed (`internal::ChargeMixer`): undamped Picard is a
/// period-2 limit cycle for polar systems and never meets `scc_tol`. Not
/// converging is an error, never a silently returned iterate.
SccSolution solve_scc(const cpu::Frame& f, const int* type,
                      const StorageReal* x, const StorageReal* y,
                      const StorageReal* z,
                      const std::vector<const DftbElementRow*>& el,
                      const std::vector<double>& hubbard_du_ev,
                      const std::vector<int>& ao0, int n_at, int n_ao,
                      int n_occ, double scc_tol, int max_scc, double mix_beta) {
    const std::size_t n = static_cast<std::size_t>(n_at);
    const auto images = lattice_images_for(f.box, el);

    linalg::RealMatrix H0 = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix S = xt::zeros<double>({n_ao, n_ao});
    // Min-image distances for Dftb3Scc (molecular kernel API); lattice-summed
    // γ is applied via gamma_pair_lat below for the potential/energy.
    std::vector<double> dist(n * n, 0.0);

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = ea.e_s;
        S(ia, ia) = 1.0;
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = ea.e_p;
                S(ia + p, ia + p) = 1.0;
            }
        }
    }
    // Lattice-summed directed SK blocks (all ordered pairs, all images).
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = 0; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double rcut = pair_rcut_A(ea, eb);
            const auto& sk = pair_sk(type[a], type[b]);
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            double rmin = 1e300;
            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12)
                    continue;
                rmin = std::min(rmin, r);
                if (r > rcut)
                    continue;
                internal::MioSketchPair{ea, eb, sk}.add_block(dx, dy, dz, r, H0,
                                                              S, ia, ib);
            }
            if (rmin < 1e299)
                dist[static_cast<std::size_t>(a) * n
                     + static_cast<std::size_t>(b)] = rmin;
        }
    }

    std::vector<double> hubbard_u(n);
    std::vector<double> hubbard_du(n);
    // Precompute lattice-summed γ and Γ for the SCC kernel.
    std::vector<double> G(n * n), Gd(n * n);
    for (int a = 0; a < n_at; ++a) {
        hubbard_u[static_cast<std::size_t>(a)] =
            el[static_cast<std::size_t>(a)]->hubbard_u;
        hubbard_du[static_cast<std::size_t>(a)] =
            hubbard_du_ev[static_cast<std::size_t>(a)];
        for (int b = 0; b < n_at; ++b) {
            G[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)] =
                gamma_pair_lat(a, b, el, x, y, z, f.box, images);
            Gd[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)] =
                gamma_charge_pair_lat(a, b, el, hubbard_du_ev, x, y, z, f.box,
                                      images);
        }
    }
    // Dftb3Scc kept for pair_energy_distance_derivative API on min-image r
    // (forces walk images explicitly and recompute one-body γ kernels).
    const internal::Dftb3Scc scc(hubbard_u, hubbard_du, dist);

    std::vector<double> dq(n, 0.0);
    std::vector<double> dq_out(n, 0.0);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix C = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector eps = xt::zeros<double>({n_ao});
    SccSolution out;
    bool converged = false;
    // Retained iterates in the Anderson charge mixer (dimensionless).
    constexpr int kSccMixDepth = 6;
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);

    auto potential_lat = [&](const std::vector<double>& q) {
        // V²_A = Σ_B γ_AB Δq_B; V³ as in Dftb3Scc::third_order_potential.
        std::vector<double> v(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double s2 = 0;
            for (int b = 0; b < n_at; ++b)
                s2 += G[static_cast<std::size_t>(a) * n
                        + static_cast<std::size_t>(b)]
                      * q[static_cast<std::size_t>(b)];
            const double dqa = q[static_cast<std::size_t>(a)];
            double s3 = 0;
            for (int b = 0; b < n_at; ++b) {
                const double dqb = q[static_cast<std::size_t>(b)];
                s3 += (2.0 / 3.0) * dqa
                      * Gd[static_cast<std::size_t>(a) * n
                           + static_cast<std::size_t>(b)]
                      * dqb;
                s3 += (1.0 / 3.0) * dqb * dqb
                      * Gd[static_cast<std::size_t>(b) * n
                           + static_cast<std::size_t>(a)];
            }
            v[static_cast<std::size_t>(a)] = s2 + s3;
        }
        return v;
    };

    for (int it = 0; it < max_scc; ++it) {
        const std::vector<double> dV = potential_lat(dq);

        linalg::RealMatrix H = H0;
        for (int a = 0; a < n_at; ++a) {
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int b = 0; b < n_at; ++b) {
                const int ib = ao0[static_cast<std::size_t>(b)];
                const int nb = el[static_cast<std::size_t>(b)]->n_ao;
                const double half = 0.5
                                    * (dV[static_cast<std::size_t>(a)]
                                       + dV[static_cast<std::size_t>(b)]);
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nb; ++j)
                        H(ia + i, ib + j) += half * S(ia + i, ib + j);
            }
        }

        auto sol = linalg::diagonalize_fock_generalized(H, S);
        eps = sol.first;
        C = sol.second;

        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

        const linalg::RealMatrix PS = xt::linalg::dot(P, S);
        double residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i)
                qa += PS(ia + i, ia + i);
            dq_out[static_cast<std::size_t>(a)] =
                qa - el[static_cast<std::size_t>(a)]->n_val;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
        }

        out.residual = residual;
        out.iterations = it + 1;
        if (residual < scc_tol) {
            dq = dq_out;
            converged = true;
            break;
        }
        dq = mixer.next(dq, dq_out);
    }

    if (!converged)
        throw std::runtime_error(
            "DFTB3: SCC did not converge in " + std::to_string(max_scc)
            + " iterations (residual " + std::to_string(out.residual)
            + " e > scc_tol " + std::to_string(scc_tol)
            + "); raise max_scc or lower mix_beta");

    for (int mu = 0; mu < n_ao; ++mu)
        for (int nu = 0; nu < n_ao; ++nu)
            out.e_h0 += P(mu, nu) * H0(mu, nu);
    // E² / E³ with lattice-summed γ / Γ.
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b) {
            const double dqa = dq[static_cast<std::size_t>(a)];
            const double dqb = dq[static_cast<std::size_t>(b)];
            out.e2 += 0.5
                      * G[static_cast<std::size_t>(a) * n
                          + static_cast<std::size_t>(b)]
                      * dqa * dqb;
            out.e3 += (1.0 / 3.0)
                      * Gd[static_cast<std::size_t>(a) * n
                           + static_cast<std::size_t>(b)]
                      * dqa * dqa * dqb;
        }
    // Lattice-summed repulsive (unordered images, factor ½ for double count).
    for (int a = 0; a < n_at; ++a) {
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = 0; b < n_at; ++b) {
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12)
                    continue;
                out.e_rep += 0.5 * pair_repulsive_eV(type[a], type[b], r);
            }
        }
    }
    out.dq = std::move(dq);
    out.C = std::move(C);
    out.eps = std::move(eps);
    out.n_occ = n_occ;
    out.scc = scc;
    return out;
}

/// Closed-form nuclear gradient at the converged SCC-DFTB3 density (fixed Δq).
///
/// `H = H⁰ + ½S(V_A+V_B)` with `V = ∂(E⁽²⁾+E⁽³⁾)/∂Δq` is exactly `∂E/∂P`
/// (Mulliken `∂Δq_A/∂P_μν = ½S_μν(δ_{μ∈A}+δ_{ν∈A})`), so the converged density
/// is stationary and the gradient is Hellmann–Feynman + Pulay with **no**
/// coupled-perturbed charge response (Gaus, Cui, Elstner, JCTC 7, 931 (2011)):
///
///   ∂E/∂R = Σ P ∂H⁰/∂R + Σ P ½(V_A+V_B) ∂S/∂R − Σ W ∂S/∂R
///           + ∂(E⁽²⁾+E⁽³⁾)/∂R|_Δq + ∂E_rep/∂R,   W_μν = Σ_o n_o ε_o C_μo C_νo
///
/// The walk mirrors `solve_scc` term by term: SK derivatives only inside
/// `pair_rcut_A`, γ/Γ and repulsive over every pair (the energy sums those
/// uncut). Every contribution is a pair force with F_b = −F_a, so ΣF = 0 to
/// round-off by construction.
void analytic_forces(int n_at, int n_ao,
                     const std::vector<const DftbElementRow*>& el,
                     const std::vector<double>& hubbard_du_ev,
                     const std::vector<int>& ao0, const int* type,
                     const StorageReal* x, const StorageReal* y,
                     const StorageReal* z, const cpu::Box& box,
                     const SccSolution& sol, AccReal* fx, AccReal* fy,
                     AccReal* fz) {
    const auto images = lattice_images_for(box, el);
    const std::size_t n = static_cast<std::size_t>(n_at);

    // Lattice-summed V for the Pulay shift (same kernel as the SCC solve).
    std::vector<double> G(n * n), Gd(n * n);
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b) {
            G[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)] =
                gamma_pair_lat(a, b, el, x, y, z, box, images);
            Gd[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)] =
                gamma_charge_pair_lat(a, b, el, hubbard_du_ev, x, y, z, box,
                                      images);
        }
    std::vector<double> dV(n, 0.0);
    for (int a = 0; a < n_at; ++a) {
        double s2 = 0;
        for (int b = 0; b < n_at; ++b)
            s2 +=
                G[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)]
                * sol.dq[static_cast<std::size_t>(b)];
        const double dqa = sol.dq[static_cast<std::size_t>(a)];
        double s3 = 0;
        for (int b = 0; b < n_at; ++b) {
            const double dqb = sol.dq[static_cast<std::size_t>(b)];
            s3 += (2.0 / 3.0) * dqa
                  * Gd[static_cast<std::size_t>(a) * n
                       + static_cast<std::size_t>(b)]
                  * dqb;
            s3 += (1.0 / 3.0) * dqb * dqb
                  * Gd[static_cast<std::size_t>(b) * n
                       + static_cast<std::size_t>(a)];
        }
        dV[static_cast<std::size_t>(a)] = s2 + s3;
    }

    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix W = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < sol.n_occ; ++o)
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                P(mu, nu) += 2.0 * sol.C(mu, o) * sol.C(nu, o);
                W(mu, nu) += 2.0 * sol.eps(o) * sol.C(mu, o) * sol.C(nu, o);
            }

    linalg::RealMatrix dH0[3] = {xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao})};
    linalg::RealMatrix dS[3] = {xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao})};

    // All ordered (a,b) and images T of B — F on a from ∂/∂R_a of each bond.
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const int na = ea.n_ao;
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        const double ua = ea.hubbard_u;
        const double uda = hubbard_du_ev[static_cast<std::size_t>(a)];
        const double dqa = sol.dq[static_cast<std::size_t>(a)];

        for (int b = 0; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const int nb = eb.n_ao;
            const double ub = eb.hubbard_u;
            const double udb = hubbard_du_ev[static_cast<std::size_t>(b)];
            const double dqb = sol.dq[static_cast<std::size_t>(b)];
            const double rcut = pair_rcut_A(ea, eb);
            const auto& sk_ab = pair_sk(type[a], type[b]);
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double shift = 0.5
                                 * (dV[static_cast<std::size_t>(a)]
                                    + dV[static_cast<std::size_t>(b)]);

            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12)
                    continue;
                const double dir[3] = {dx / r, dy / r, dz / r};
                double F_a[3] = {0, 0, 0};

                if (r <= rcut) {
                    for (int k = 0; k < 3; ++k) {
                        dH0[k] = xt::zeros<double>({n_ao, n_ao});
                        dS[k] = xt::zeros<double>({n_ao, n_ao});
                    }
                    // Only the A→B directed block for this ordered (a,b,T).
                    // The (b,a,−T) image is visited separately in the a↔b walk.
                    internal::MioSketchPair{ea, eb, sk_ab}.set_block_derivative(
                        dx, dy, dz, r, dH0, dS, ia, ib);
                    for (int k = 0; k < 3; ++k) {
                        double acc = 0;
                        for (int i = 0; i < na; ++i)
                            for (int j = 0; j < nb; ++j) {
                                const double p = P(ia + i, ib + j);
                                const double w = W(ia + i, ib + j);
                                acc += p
                                           * (dH0[k](ia + i, ib + j)
                                              + shift * dS[k](ia + i, ib + j))
                                       - w * dS[k](ia + i, ib + j);
                            }
                        // R = R_b+T−R_a ⇒ ∂/∂R_a = −∂/∂R ⇒ F_a = +acc.
                        F_a[k] += acc;
                    }
                }

                // Fixed-Δq ∂(E²+E³)/∂r + repulsive for this ordered image.
                // Energy uses ½ Σ_{a,b,T} for E²/E_rep and ⅓ Σ for E³; the
                // partner (b,a,−T) supplies the matching half.
                const double dE_dr =
                    0.5 * dqa * dqb * dgamma1_dr(r, ua, ub)
                    + (1.0 / 6.0)
                          * (dqa * dqa * dqb * gamma_charge1_dr(r, ua, ub, uda)
                             + dqb * dqb * dqa
                                   * gamma_charge1_dr(r, ub, ua, udb))
                    + 0.5 * pair_repulsive_dE_dr(type[a], type[b], r);
                // ∂r/∂R_a = −r̂ ⇒ F_a = +(dE_dr) r̂; F_b = −F_a.
                // Self-images (a=b): F_a and −F_a cancel — correct, bond T is
                // nuclear-coordinate independent (DFTB2 same).
                for (int k = 0; k < 3; ++k)
                    F_a[k] += dE_dr * dir[k];

                fx[a] += static_cast<AccReal>(F_a[0]);
                fy[a] += static_cast<AccReal>(F_a[1]);
                fz[a] += static_cast<AccReal>(F_a[2]);
                fx[b] -= static_cast<AccReal>(F_a[0]);
                fy[b] -= static_cast<AccReal>(F_a[1]);
                fz[b] -= static_cast<AccReal>(F_a[2]);
            }
        }
    }
}

} // namespace

namespace {

constexpr const char* kDftbHubbardDerivHowto =
    "d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;";
constexpr const char* kDftbSkSetHowto =
    "DFTB3 will not pick an SK set for you — declare "
    "d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;";

} // namespace

void DFTB3::build(const BuildContext& /*ctx*/) {}

void DFTB3::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("DFTB3: empty frame");
    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument("DFTB3: mix_beta must be in (0, 1]");

    switch (sk_set) {
    case DftbSkSet::Unset:
        throw std::invalid_argument(
            std::string("DFTB3::launch: sk_set is unset — ") + kDftbSkSetHowto);
    case DftbSkSet::ThreeOb:
        throw std::runtime_error(
            "DFTB3::launch: sk_set = ThreeOb is not implemented — "
            "dftb-gamma-h-damping");
    case DftbSkSet::MioSketch:
        break;
    }

    const auto type = f.span(kAtom_type);
    // U^d completeness before element_row so unsupported Z (e.g. 9) still
    // reports hubbard_du_ha rather than "unsupported element".
    std::vector<double> hubbard_du_ev(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        const int z = type[a];
        if (z < 0 || z >= static_cast<int>(hubbard_du_ha.size())
            || hubbard_du_ha[static_cast<std::size_t>(z)] == 0.0
            || !std::isfinite(hubbard_du_ha[static_cast<std::size_t>(z)]))
            throw std::invalid_argument(
                std::string("DFTB3::launch: hubbard_du_ha is unset for Z=")
                + std::to_string(z) + " — " + kDftbHubbardDerivHowto);
        hubbard_du_ev[static_cast<std::size_t>(a)] =
            hubbard_du_ha[static_cast<std::size_t>(z)] * units::kHartreeInEv;
    }

    if (sk_set == DftbSkSet::MioSketch && !warned_mixed_set_) {
        ATV_WARN("[dftb3] 3ob-3-1 README: 3ob supersedes the MIO parameter "
                 "set; MioSketch SK + 3ob HubbardDerivs is a mixed-set pairing "
                 "and must not be labelled oracle: dftb+");
        warned_mixed_set_ = true;
    }

    // Read-only: the closed-form gradient never displaces the geometry (the
    // deleted FD path did, 6N times per force evaluation).
    const StorageReal* x = f.span(kAtom_x);
    const StorageReal* y = f.span(kAtom_y);
    const StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);

    std::vector<const DftbElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = 0;
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB3: no basis functions");
    if (std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "DFTB3: odd/non-integer valence electron count not supported");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    const SccSolution sol =
        solve_scc(f, type.data(), x, y, z, el, hubbard_du_ev, ao0, n_at, n_ao,
                  n_occ, scc_tol, max_scc, mix_beta);
    e2_ = sol.e2;
    e3_ = sol.e3;
    scc_residual_ = sol.residual;
    scc_iterations_ = sol.iterations;
    // E = Tr(ρH⁰) + E² + E³ + E_rep. The band energy is NOT the starting
    // point: Tr(ρH) = Tr(ρH⁰) + Σ_A V_A q_A carries the GROSS Mulliken
    // population, so subtracting a double-counting built from Δq alone
    // leaves a spurious Σ_A V_A q⁰_A that is linear in Δq.
    energy_ = sol.energy();
    energy_meta += energy_;

    if (with_forces)
        analytic_forces(n_at, n_ao, el, hubbard_du_ev, ao0, type.data(), x, y,
                        z, f.box, sol, fx, fy, fz);
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
