/// @file src/cpu/semiempirical/dftb_gamma_lattice.cpp
/// @brief Ewald-summed SCC charge kernel γ — see the header for the physics.

#include "atomiverse/cpu/semiempirical/internal/dftb_gamma_lattice.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/longrange/ewald_sum.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_gamma.h"

// Column-major GEMM. Declared here rather than pulled in through
// xtensor-blas for the same reason `eigensolve.h` spells out its LAPACK
// bindings by hand: this is one call on raw buffers and it must accumulate
// (beta = 1) straight into γ instead of materialising an N×N temporary.
extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

namespace {

/// Elstner short-range piece S(τ_A, τ_B, R) [eV] at R [Å], U [eV].
///
/// γ = C/R − S; this returns S alone so the 1/R half can go to Ewald.
double short_range_eV(double r_A, double ua_eV, double ub_eV) {
    return dftb_gamma::exp_gamma_au(r_A * units::kAngstromInBohr,
                                    ua_eV * units::kEvInHartree,
                                    ub_eV * units::kEvInHartree)
           * units::kHartreeInEv;
}

/// dS/dr [eV/Å] — the partner of `short_range_eV`.
double d_short_range_dr_eV(double r_A, double ua_eV, double ub_eV) {
    return dftb_gamma::exp_gamma_prime_au(r_A * units::kAngstromInBohr,
                                          ua_eV * units::kEvInHartree,
                                          ub_eV * units::kEvInHartree)
           * units::kAngstromInBohr * units::kHartreeInEv;
}

/// Absolute value below which the short-range piece S is dropped [eV].
///
/// S decays as \f$e^{-3.2 U r}\f$ times a polynomial; rather than bound that
/// analytically the cutoff is found by walking the real function outwards, so
/// the prefactor is accounted for exactly.
constexpr double kShortRangeTolEv = 1.0e-12;

/// Range beyond which S(τ,τ,r) is below `kShortRangeTolEv` [Å].
///
/// Evaluated at the **softest** atom present (smallest U ⇒ slowest decay), so
/// one radius bounds every pair in the system.
double short_range_cutoff(double u_min_eV) {
    constexpr double kStep = 0.25;    // Å
    constexpr double kCeiling = 60.0; // Å — S is long dead well before this
    double r = kStep;
    while (r < kCeiling
           && std::fabs(short_range_eV(r, u_min_eV, u_min_eV))
                  > kShortRangeTolEv)
        r += kStep;
    return r;
}

/// Integer image half-widths covering @p cutoff along every cell axis.
void image_extent(const Mat3& cell, double cutoff, int nmax[3]) {
    for (int d = 0; d < 3; ++d) {
        const double len =
            std::sqrt(cell(d, 0) * cell(d, 0) + cell(d, 1) * cell(d, 1)
                      + cell(d, 2) * cell(d, 2));
        if (!(len > 0.0))
            throw std::runtime_error(
                "GammaLattice: cell axis " + std::to_string(d)
                + " has zero length; a periodic γ needs a real cell");
        nmax[d] = static_cast<int>(std::ceil(cutoff / len)) + 1;
    }
}

} // namespace

GammaLattice::GammaLattice(int n_at, const double* u_eV, const double* x,
                           const double* y, const double* z,
                           const cpu::Box& box, bool periodic,
                           double real_cutoff, double accuracy)
    : n_(n_at), periodic_(periodic) {
    if (n_at < 1)
        throw std::invalid_argument("GammaLattice: n_at must be >= 1");
    if (!(real_cutoff > 0.0))
        throw std::invalid_argument("GammaLattice: real_cutoff must be > 0");
    if (!(accuracy > 0.0 && accuracy < 1.0))
        throw std::invalid_argument("GammaLattice: accuracy must be in (0, 1)");

    const std::size_t n = static_cast<std::size_t>(n_at);
    x_.assign(x, x + n);
    y_.assign(y, y + n);
    z_.assign(z, z + n);
    u_.assign(u_eV, u_eV + n);
    real_cutoff_ = real_cutoff;
    g_.assign(n * n, 0.0);
    auto at = [&](std::size_t a, std::size_t b) -> double& {
        return g_[a * n + b];
    };

    // ── Cluster: no images, γ is the bare pair function. ─────────────────
    if (!periodic_) {
        for (std::size_t a = 0; a < n; ++a) {
            at(a, a) = u_eV[a];
            for (std::size_t b = a + 1; b < n; ++b) {
                const double dx = x[b] - x[a];
                const double dy = y[b] - y[a];
                const double dz = z[b] - z[a];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double g = dftb_gamma::gamma_eV(r, u_eV[a], u_eV[b]);
                at(a, b) = g;
                at(b, a) = g;
            }
        }
        return;
    }

    // ── Periodic: Ewald 1/R + real-space short range. ────────────────────
    const double volume = box.volume();
    if (!(volume > 0.0))
        throw std::runtime_error(
            "GammaLattice: periodic γ needs a cell of positive volume");

    double u_min = u_eV[0];
    for (std::size_t a = 1; a < n; ++a)
        u_min = std::min(u_min, u_eV[a]);
    if (!(u_min > 0.0))
        throw std::runtime_error(
            "GammaLattice: Hubbard U must be positive for every atom");
    short_cutoff_ = short_range_cutoff(u_min);

    volume_ = volume;
    cell_ = box.cell();
    alpha_ = longrange::EwaldSum::recommend_alpha(real_cutoff, accuracy);
    ewald_ = longrange::EwaldSum{alpha_, accuracy};
    ewald_.rebuild(box.inv_cell());
    const longrange::EwaldSum& ewald = ewald_;

    const double C = units::coulomb;
    const double walk_cutoff = std::max(short_cutoff_, real_cutoff);
    const Mat3& cell = cell_;
    image_extent(cell, walk_cutoff, nmax_);

    // Real space: erfc(αr)/r over every image inside real_cutoff, minus the
    // exponential short-range piece inside short_cutoff. (A,A,T=0) excluded.
    //
    // Atom A is the outermost loop so that a thread owns whole rows of γ and
    // the accumulation needs no synchronisation; the images sit inside.
    const double short_cut2 = short_cutoff_ * short_cutoff_;
    const double real_cut2 = real_cutoff * real_cutoff;
    const double walk_cut2 = std::max(short_cut2, real_cut2);
    const int nat = n_at;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ai = 0; ai < nat; ++ai) {
        const std::size_t a = static_cast<std::size_t>(ai);
        double* row = g_.data() + a * n;
        for (int n0 = -nmax_[0]; n0 <= nmax_[0]; ++n0)
            for (int n1 = -nmax_[1]; n1 <= nmax_[1]; ++n1)
                for (int n2 = -nmax_[2]; n2 <= nmax_[2]; ++n2) {
                    const double tx = n0 * cell(0, 0) + n1 * cell(1, 0)
                                      + n2 * cell(2, 0) - x[a];
                    const double ty = n0 * cell(0, 1) + n1 * cell(1, 1)
                                      + n2 * cell(2, 1) - y[a];
                    const double tz = n0 * cell(0, 2) + n1 * cell(1, 2)
                                      + n2 * cell(2, 2) - z[a];
                    for (std::size_t b = 0; b < n; ++b) {
                        const double dx = x[b] + tx;
                        const double dy = y[b] + ty;
                        const double dz = z[b] + tz;
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < 1e-24 || r2 > walk_cut2)
                            continue; // 1e-24 is the excluded (A,A,T=0) term
                        const double r = std::sqrt(r2);
                        double g = 0.0;
                        if (r2 <= real_cut2)
                            g += C * std::erfc(alpha_ * r) / r;
                        if (r2 <= short_cut2)
                            g -= short_range_eV(r, u_eV[a], u_eV[b]);
                        row[b] += g;
                    }
                }
    }

    // Reciprocal space. E_recip = C(4π/V) Σ_k w_k |S(k)|² over the positive
    // hemisphere; as a symmetric quadratic form in Δq that is 8π/V per pair.
    //
    // cos(k·R_ab) = cos(k·R_a)cos(k·R_b) + sin(k·R_a)sin(k·R_b) turns the
    // N²K sum into two rank-K updates, so the assembly is BLAS3 rather than a
    // scalar triple loop — the difference between minutes and a second at
    // N ≈ 1000. k is blocked so the trig tables stay a few MB.
    const std::size_t nk = ewald.size();
    const double* kx = ewald.kx();
    const double* ky = ewald.ky();
    const double* kz = ewald.kz();
    const double* kw = ewald.kw();
    const double recip_pref = C * 8.0 * units::pi / volume;
    {
        constexpr std::size_t kBlock = 256;
        const int ni = static_cast<int>(n);
        // Column-major (k-fastest) so the GEMM is Aᵀ·B on Kb×N operands.
        std::vector<double> cs(kBlock * n), sn(kBlock * n);
        std::vector<double> csw(kBlock * n), snw(kBlock * n);
        const double one = 1.0;
        for (std::size_t k0 = 0; k0 < nk; k0 += kBlock) {
            const std::size_t kb = std::min(kBlock, nk - k0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int ai = 0; ai < ni; ++ai) {
                const std::size_t a = static_cast<std::size_t>(ai);
                for (std::size_t j = 0; j < kb; ++j) {
                    const std::size_t k = k0 + j;
                    const double ph =
                        kx[k] * x[a] + ky[k] * y[a] + kz[k] * z[a];
                    const double c = std::cos(ph), sv = std::sin(ph);
                    cs[a * kb + j] = c;
                    sn[a * kb + j] = sv;
                    csw[a * kb + j] = recip_pref * kw[k] * c;
                    snw[a * kb + j] = recip_pref * kw[k] * sv;
                }
            }
            const int kbi = static_cast<int>(kb);
            dgemm_("T", "N", &ni, &ni, &kbi, &one, csw.data(), &kbi, cs.data(),
                   &kbi, &one, g_.data(), &ni);
            dgemm_("T", "N", &ni, &ni, &kbi, &one, snw.data(), &kbi, sn.data(),
                   &kbi, &one, g_.data(), &ni);
        }
    }

    // Neutralising background (survives only for a charged cell) and the
    // Ewald self term, then the DFTB on-site Hubbard U.
    const double background = -C * units::pi / (alpha_ * alpha_ * volume);
    const double self = -C * 2.0 * alpha_ / std::sqrt(units::pi);
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b)
            at(a, b) += background;
        at(a, a) += self + u_eV[a];
    }
}

void GammaLattice::potential(const double* dq, double* out) const {
    const std::size_t n = static_cast<std::size_t>(n_);
    for (std::size_t a = 0; a < n; ++a) {
        const double* ga = g_.data() + a * n;
        double v = 0.0;
        for (std::size_t b = 0; b < n; ++b)
            v += ga[b] * dq[b];
        out[a] = v;
    }
}

double GammaLattice::energy(const double* dq) const {
    const std::size_t n = static_cast<std::size_t>(n_);
    double e = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
        const double* ga = g_.data() + a * n;
        double v = 0.0;
        for (std::size_t b = 0; b < n; ++b)
            v += ga[b] * dq[b];
        e += dq[a] * v;
    }
    return 0.5 * e;
}

void GammaLattice::add_gradient(const double* dq, double* fx, double* fy,
                                double* fz, double* dE_de) const {
    const std::size_t n = static_cast<std::size_t>(n_);
    const bool want_f = fx != nullptr && fy != nullptr && fz != nullptr;
    if (!want_f && dE_de == nullptr)
        return;
    const double C = units::coulomb;

    // One cluster bond (a,b): force on a, equal and opposite on b, plus the
    // pair's ∂E/∂ε. The lattice walk below does not use this — it accumulates
    // per-atom so it can be threaded.
    auto accumulate = [&](std::size_t a, std::size_t b, const double d[3],
                          double r, double dgdr, double weight) {
        const double s = weight * dq[a] * dq[b] * dgdr / r;
        const double f[3] = {s * d[0], s * d[1], s * d[2]};
        if (want_f && a != b) {
            fx[a] += f[0];
            fy[a] += f[1];
            fz[a] += f[2];
            fx[b] -= f[0];
            fy[b] -= f[1];
            fz[b] -= f[2];
        }
        if (dE_de) {
            dE_de[0] += f[0] * d[0];
            dE_de[1] += f[1] * d[1];
            dE_de[2] += f[2] * d[2];
            dE_de[3] += f[0] * d[1] + f[1] * d[0];
            dE_de[4] += f[0] * d[2] + f[2] * d[0];
            dE_de[5] += f[1] * d[2] + f[2] * d[1];
        }
    };

    // ── Cluster: the bare pair derivative, no images. ────────────────────
    if (!periodic_) {
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b) {
                const double d[3] = {x_[b] - x_[a], y_[b] - y_[a],
                                     z_[b] - z_[a]};
                const double r =
                    std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                if (r < 1e-12)
                    continue;
                accumulate(a, b, d, r,
                           dftb_gamma::dgamma_dr_eV(r, u_[a], u_[b]), 1.0);
            }
        return;
    }

    // ── Real space: the same erfc + S walk the matrix was built from. ────
    //
    // Ordered over **all** b (not b > a) so atom A owns its whole force and a
    // thread never touches another thread's accumulator. The virial then
    // carries ½ per ordered term, since (a,b,T) and (b,a,−T) contribute
    // equally; the force carries 1, because both halves of ∂γ_ab/∂R_a do.
    // The A-with-own-image terms cancel pairwise in the force (T against −T)
    // and survive in the virial, which is exactly right: that bond is T alone.
    const double short_cut2 = short_cutoff_ * short_cutoff_;
    const double real_cut2 = real_cutoff_ * real_cutoff_;
    const double walk_cut2 = std::max(short_cut2, real_cut2);
    const double two_alpha_over_sqrtpi = 2.0 * alpha_ / std::sqrt(units::pi);
    const int nat = n_;
    double v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : v0, v1, v2, v3, v4, v5)
#endif
    for (int ai = 0; ai < nat; ++ai) {
        const std::size_t a = static_cast<std::size_t>(ai);
        double fa[3] = {0, 0, 0};
        for (int n0 = -nmax_[0]; n0 <= nmax_[0]; ++n0)
            for (int n1 = -nmax_[1]; n1 <= nmax_[1]; ++n1)
                for (int n2 = -nmax_[2]; n2 <= nmax_[2]; ++n2) {
                    const double tx = n0 * cell_(0, 0) + n1 * cell_(1, 0)
                                      + n2 * cell_(2, 0) - x_[a];
                    const double ty = n0 * cell_(0, 1) + n1 * cell_(1, 1)
                                      + n2 * cell_(2, 1) - y_[a];
                    const double tz = n0 * cell_(0, 2) + n1 * cell_(1, 2)
                                      + n2 * cell_(2, 2) - z_[a];
                    for (std::size_t b = 0; b < n; ++b) {
                        const double d[3] = {x_[b] + tx, y_[b] + ty,
                                             z_[b] + tz};
                        const double r2 =
                            d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if (r2 < 1e-24 || r2 > walk_cut2)
                            continue;
                        const double r = std::sqrt(r2);
                        double dgdr = 0.0;
                        if (r2 <= real_cut2)
                            dgdr -=
                                C
                                * (std::erfc(alpha_ * r) / r2
                                   + two_alpha_over_sqrtpi
                                         * std::exp(-alpha_ * alpha_ * r2) / r);
                        if (r2 <= short_cut2)
                            dgdr -= d_short_range_dr_eV(r, u_[a], u_[b]);
                        const double s = dq[a] * dq[b] * dgdr / r;
                        const double f[3] = {s * d[0], s * d[1], s * d[2]};
                        fa[0] += f[0];
                        fa[1] += f[1];
                        fa[2] += f[2];
                        v0 += 0.5 * f[0] * d[0];
                        v1 += 0.5 * f[1] * d[1];
                        v2 += 0.5 * f[2] * d[2];
                        v3 += 0.5 * (f[0] * d[1] + f[1] * d[0]);
                        v4 += 0.5 * (f[0] * d[2] + f[2] * d[0]);
                        v5 += 0.5 * (f[1] * d[2] + f[2] * d[1]);
                    }
                }
        if (want_f) {
            fx[a] += fa[0];
            fy[a] += fa[1];
            fz[a] += fa[2];
        }
    }
    if (dE_de) {
        dE_de[0] += v0;
        dE_de[1] += v1;
        dE_de[2] += v2;
        dE_de[3] += v3;
        dE_de[4] += v4;
        dE_de[5] += v5;
    }

    // ── Reciprocal space. ────────────────────────────────────────────────
    const std::size_t nk = ewald_.size();
    const double* kx = ewald_.kx();
    const double* ky = ewald_.ky();
    const double* kz = ewald_.kz();
    const double* kw = ewald_.kw();
    const double* kvt = ewald_.kvterm();
    const double e_pref = C * 4.0 * units::pi / volume_;
    for (std::size_t k = 0; k < nk; ++k) {
        double sr = 0.0, si = 0.0;
        for (std::size_t a = 0; a < n; ++a) {
            const double ph = kx[k] * x_[a] + ky[k] * y_[a] + kz[k] * z_[a];
            sr += dq[a] * std::cos(ph);
            si += dq[a] * std::sin(ph);
        }
        if (want_f) {
            // F_A = −(8πC/V) Δq_A Σ_k w_k [S_i cos(k·R_A) − S_r sin(k·R_A)] k
            const double pref = -2.0 * e_pref * kw[k];
            for (std::size_t a = 0; a < n; ++a) {
                const double ph = kx[k] * x_[a] + ky[k] * y_[a] + kz[k] * z_[a];
                const double c =
                    pref * dq[a] * (si * std::cos(ph) - sr * std::sin(ph));
                fx[a] += c * kx[k];
                fy[a] += c * ky[k];
                fz[a] += c * kz[k];
            }
        }
        if (dE_de) {
            // E_k ∝ (1/V)·w(k²)·|S|², k·R is strain-invariant, ∂k²/∂ε_ij =
            // −2k_i k_j and ∂(1/V)/∂ε_ij = −δ_ij/V, giving
            //   ∂E_k/∂ε_ij = −U_k (δ_ij + kvterm_k k_i k_j).
            // Note the **minus**: `EwaldSum::reciprocal_energy_virial` returns
            // the same bracket without it, because that one is a LAMMPS-style
            // virial (−∂E/∂ε) while `dE_de` here is +∂E/∂ε. Getting this
            // backwards is invisible on any homonuclear system, where Δq ≡ 0
            // kills the whole γ stress.
            const double uk = e_pref * kw[k] * (sr * sr + si * si);
            dE_de[0] -= uk * (1.0 + kvt[k] * kx[k] * kx[k]);
            dE_de[1] -= uk * (1.0 + kvt[k] * ky[k] * ky[k]);
            dE_de[2] -= uk * (1.0 + kvt[k] * kz[k] * kz[k]);
            dE_de[3] -= 2.0 * uk * kvt[k] * kx[k] * ky[k];
            dE_de[4] -= 2.0 * uk * kvt[k] * kx[k] * kz[k];
            dE_de[5] -= 2.0 * uk * kvt[k] * ky[k] * kz[k];
        }
    }

    // ── Neutralising background: E ∝ 1/V, so ∂E/∂ε = −E δ_ij. Zero for a
    //    neutral cell; the Ewald self term has no cell or position dependence
    //    at all and so contributes to neither. ─────────────────────────────
    if (dE_de) {
        double q_total = 0.0;
        for (std::size_t a = 0; a < n; ++a)
            q_total += dq[a];
        const double e_bg = -C * units::pi / (2.0 * volume_ * alpha_ * alpha_)
                            * q_total * q_total;
        for (int v = 0; v < 3; ++v)
            dE_de[v] -= e_bg;
    }
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
