/// @file src/cpu/longrange/ewald_sum.cpp
/// @brief Reciprocal-space Ewald kernels (see the header).
#include "atomiverse/cpu/longrange/ewald_sum.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace longrange {

namespace {

/// Atoms per stack block of the kernels. S(k) is summed block by block in
/// atom order, so the block (not the team) fixes its rounding.
constexpr int kBlock = 256;

/// Threads worth opening for `pairs` (k-vector, atom) pairs.
int team_for(std::size_t pairs) {
#ifdef _OPENMP
    const std::size_t want = pairs / EwaldSum::kPairsPerThread;
    const auto most = static_cast<std::size_t>(omp_get_max_threads());
    return static_cast<int>(std::clamp<std::size_t>(want, 1, most));
#else
    (void)pairs;
    return 1;
#endif
}

/// Rows [begin, end) of thread `rank` out of `team` over `n` atoms.
void team_range(int n, int rank, int team, int& begin, int& end) {
    begin = static_cast<int>(static_cast<long long>(n) * rank / team);
    end = static_cast<int>(static_cast<long long>(n) * (rank + 1) / team);
}

/// exp(i (mx b₀ + my b₁)·r_j) for atoms [j0, j0 + jn), times q_j when `q`
/// is given. A negative power is the conjugate of the tabulated one.
void column_phase(const EwaldPhases& ph, int mx, int my, int j0, int jn,
                  const StorageReal* q, double* __restrict pr,
                  double* __restrict pi) {
    const double* __restrict cx = ph.cos_row(0, std::abs(mx)) + j0;
    const double* __restrict sx = ph.sin_row(0, std::abs(mx)) + j0;
    const double* __restrict cy = ph.cos_row(1, std::abs(my)) + j0;
    const double* __restrict sy = ph.sin_row(1, std::abs(my)) + j0;
    const double gx = mx < 0 ? -1.0 : 1.0;
    const double gy = my < 0 ? -1.0 : 1.0;
    if (q == nullptr) {
#pragma omp simd
        for (int j = 0; j < jn; ++j) {
            const double ux = gx * sx[j], uy = gy * sy[j];
            pr[j] = cx[j] * cy[j] - ux * uy;
            pi[j] = cx[j] * uy + ux * cy[j];
        }
        return;
    }
    const StorageReal* __restrict qj = q + j0;
#pragma omp simd
    for (int j = 0; j < jn; ++j) {
        const double ux = gx * sx[j], uy = gy * sy[j];
        pr[j] = qj[j] * (cx[j] * cy[j] - ux * uy);
        pi[j] = qj[j] * (cx[j] * uy + ux * cy[j]);
    }
}

} // namespace

bool EwaldSum::rebuild(const Mat3& inv_cell) {
    // Reciprocal basis b_i = 2π · (row i of cell⁻¹); g = Σ m_i b_i.
    const double tp = 2.0 * ATV::units::pi;
    for (int i = 0; i < 3; ++i) {
        b_[i][0] = tp * inv_cell(i, 0);
        b_[i][1] = tp * inv_cell(i, 1);
        b_[i][2] = tp * inv_cell(i, 2);
    }
    const auto& b = b_;
    const Mat3 cell = inv_cell.inverse();
    const double gcut = 2.0 * alpha_ * std::sqrt(-std::log(accuracy_));
    const double gcut2 = gcut * gcut;
    int kmax[3];
    for (int i = 0; i < 3; ++i) {
        // m_i = a_i dot G / (2*pi): reciprocal vector norms alone
        // underestimate the integer range when basis vectors cancel.
        const double a_norm =
            std::sqrt(cell(0, i) * cell(0, i) + cell(1, i) * cell(1, i)
                      + cell(2, i) * cell(2, i));
        kmax[i] = std::max(1, static_cast<int>(std::ceil(gcut * a_norm / tp)));
    }

    const std::size_t max_vectors =
        static_cast<std::size_t>(2 * kmax[0] + 1)
            * static_cast<std::size_t>(2 * kmax[1] + 1)
            * static_cast<std::size_t>(2 * kmax[2] + 1)
        - 1;
    const std::size_t max_columns = static_cast<std::size_t>(2 * kmax[0] + 1)
                                    * static_cast<std::size_t>(2 * kmax[1] + 1);
    const bool capacity_grows =
        kx_.capacity() < max_vectors || ky_.capacity() < max_vectors
        || kz_.capacity() < max_vectors || kw_.capacity() < max_vectors
        || kvterm_.capacity() < max_vectors
        || columns_.capacity() < max_columns;
    if (capacity_grows) {
        kx_.reserve(max_vectors);
        ky_.reserve(max_vectors);
        kz_.reserve(max_vectors);
        kw_.reserve(max_vectors);
        kvterm_.reserve(max_vectors);
        columns_.reserve(max_columns);
    }

    const double inv4a2 = 1.0 / (4.0 * alpha_ * alpha_);
    kx_.clear();
    ky_.clear();
    kz_.clear();
    kw_.clear();
    kvterm_.clear();
    columns_.clear();
    mmax_ = {0, 0, 0};
    // Positive hemisphere only (LAMMPS Ewald convention). Paired with
    // prefactor 4π/V in energy/forces (not 2π/V over the full sphere).
    for (int mx = -kmax[0]; mx <= kmax[0]; ++mx)
        for (int my = -kmax[1]; my <= kmax[1]; ++my)
            for (int mz = -kmax[2]; mz <= kmax[2]; ++mz) {
                if (mx == 0 && my == 0 && mz == 0)
                    continue;
                if (mz < 0)
                    continue;
                if (mz == 0 && my < 0)
                    continue;
                if (mz == 0 && my == 0 && mx < 0)
                    continue;
                const double gx = mx * b[0][0] + my * b[1][0] + mz * b[2][0];
                const double gy = mx * b[0][1] + my * b[1][1] + mz * b[2][1];
                const double gz = mx * b[0][2] + my * b[1][2] + mz * b[2][2];
                const double g2 = gx * gx + gy * gy + gz * gz;
                if (g2 > gcut2)
                    continue;
                const int k = static_cast<int>(kx_.size());
                kx_.push_back(gx);
                ky_.push_back(gy);
                kz_.push_back(gz);
                kw_.push_back(std::exp(-g2 * inv4a2) / g2);
                kvterm_.push_back(-2.0 * (1.0 / g2 + inv4a2));
                // The kept m₂ of one (m₀, m₁) are consecutive (the sphere
                // is convex); a gap would still open a new column.
                if (columns_.empty() || columns_.back().mx != mx
                    || columns_.back().my != my
                    || columns_.back().mz0 + (k - columns_.back().begin) != mz)
                    columns_.push_back({mx, my, mz, k, k});
                ++columns_.back().end;
                mmax_[0] = std::max(mmax_[0], std::abs(mx));
                mmax_[1] = std::max(mmax_[1], std::abs(my));
                mmax_[2] = std::max(mmax_[2], mz);
            }
    return capacity_grows;
}

void EwaldSum::phases(const StorageReal* x, const StorageReal* y,
                      const StorageReal* z, int n, EwaldPhases& out) const {
    out.n = n;
    out.mmax = mmax_;
    for (int a = 0; a < 3; ++a) {
        const std::size_t rows = static_cast<std::size_t>(mmax_[a]) + 1;
        out.re[a].resize(rows * static_cast<std::size_t>(n));
        out.im[a].resize(rows * static_cast<std::size_t>(n));
    }
    const int team = team_for(static_cast<std::size_t>(n)
                              * (mmax_[0] + mmax_[1] + mmax_[2] + 3));
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int rank = 0, size = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num();
        size = omp_get_num_threads();
#endif
        int j0, j1;
        team_range(n, rank, size, j0, j1);
        for (int a = 0; a < 3; ++a) {
            double* __restrict re = out.re[a].data();
            double* __restrict im = out.im[a].data();
            const double ba0 = b_[a][0], ba1 = b_[a][1], ba2 = b_[a][2];
            for (int j = j0; j < j1; ++j) {
                const double phase = ba0 * x[j] + ba1 * y[j] + ba2 * z[j];
                re[j] = 1.0;
                im[j] = 0.0;
                if (mmax_[a] > 0) {
                    re[n + j] = std::cos(phase);
                    im[n + j] = std::sin(phase);
                }
            }
            // exp(i m θ) = exp(i (m−1) θ) · exp(i θ).
            const double* __restrict c1 = re + n;
            const double* __restrict s1 = im + n;
            for (int m = 2; m <= mmax_[a]; ++m) {
                const double* __restrict cp = re + (m - 1) * std::size_t(n);
                const double* __restrict sp = im + (m - 1) * std::size_t(n);
                double* __restrict cm = re + m * std::size_t(n);
                double* __restrict sm = im + m * std::size_t(n);
#pragma omp simd
                for (int j = j0; j < j1; ++j) {
                    cm[j] = cp[j] * c1[j] - sp[j] * s1[j];
                    sm[j] = sp[j] * c1[j] + cp[j] * s1[j];
                }
            }
        }
    }
}

void EwaldSum::structure_factor(const EwaldPhases& ph, const StorageReal* q,
                                int n, double* sr, double* si) const {
    const int n_columns = static_cast<int>(columns_.size());
    const int team =
        team_for(static_cast<std::size_t>(n) * (kx_.size() + columns_.size()));
#pragma omp parallel for num_threads(team) if (team > 1) schedule(dynamic, 1)
    for (int c = 0; c < n_columns; ++c) {
        const Column col = columns_[c];
        for (int k = col.begin; k < col.end; ++k) {
            sr[k] = 0.0;
            si[k] = 0.0;
        }
        alignas(64) double qr[kBlock];
        alignas(64) double qi[kBlock];
        for (int j0 = 0; j0 < n; j0 += kBlock) {
            const int jn = std::min(kBlock, n - j0);
            column_phase(ph, col.mx, col.my, j0, jn, q, qr, qi);
            for (int k = col.begin; k < col.end; ++k) {
                const int mz = col.mz0 + (k - col.begin);
                const double* __restrict cz = ph.cos_row(2, mz) + j0;
                const double* __restrict sz = ph.sin_row(2, mz) + j0;
                double re = 0.0, im = 0.0;
#pragma omp simd reduction(+ : re, im)
                for (int j = 0; j < jn; ++j) {
                    re += qr[j] * cz[j] - qi[j] * sz[j];
                    im += qr[j] * sz[j] + qi[j] * cz[j];
                }
                sr[k] += re;
                si[k] += im;
            }
        }
    }
}

// F_j = 2K q_j Σ_k A_k Im(e^{ik·r_j} S̄_k) k, with k = (m₀b₀ + m₁b₁) + m₂b₂
// split per column: Σ_k t_k k = (m₀b₀ + m₁b₁) Σ t_k + b₂ Σ m₂ t_k.
void EwaldSum::add_forces(const EwaldPhases& ph, const StorageReal* q, int n,
                          double volume, const double* sr, const double* si,
                          AccReal* fx, AccReal* fy, AccReal* fz) const {
    // Half-sphere: energy prefactor is 4π k_e / V; force factor uses 2·ug.
    const double K2 = 2.0 * (qqrd2e_ * 4.0 * ATV::units::pi / volume);
    const double* __restrict kw = kw_.data();
    const int team =
        team_for(static_cast<std::size_t>(n) * (kx_.size() + columns_.size()));
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int rank = 0, size = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num();
        size = omp_get_num_threads();
#endif
        int i0, i1;
        team_range(n, rank, size, i0, i1);
        alignas(64) double pr[kBlock], pi[kBlock];
        alignas(64) double t1[kBlock], t2[kBlock];
        alignas(64) double ax[kBlock], ay[kBlock], az[kBlock];
        for (int j0 = i0; j0 < i1; j0 += kBlock) {
            const int jn = std::min(kBlock, i1 - j0);
            std::fill_n(ax, jn, 0.0);
            std::fill_n(ay, jn, 0.0);
            std::fill_n(az, jn, 0.0);
            for (const Column& col : columns_) {
                column_phase(ph, col.mx, col.my, j0, jn, nullptr, pr, pi);
                std::fill_n(t1, jn, 0.0);
                std::fill_n(t2, jn, 0.0);
                for (int k = col.begin; k < col.end; ++k) {
                    const int mz = col.mz0 + (k - col.begin);
                    const double* __restrict cz = ph.cos_row(2, mz) + j0;
                    const double* __restrict sz = ph.sin_row(2, mz) + j0;
                    const double ar = kw[k] * sr[k], ai = kw[k] * si[k];
                    const double m2 = mz;
                    // t = A (sin(k·r) S_r − cos(k·r) S_i)
#pragma omp simd
                    for (int j = 0; j < jn; ++j) {
                        const double t = pr[j] * (ar * sz[j] - ai * cz[j])
                                         + pi[j] * (ar * cz[j] + ai * sz[j]);
                        t1[j] += t;
                        t2[j] += m2 * t;
                    }
                }
                const double g0 = col.mx * b_[0][0] + col.my * b_[1][0];
                const double g1 = col.mx * b_[0][1] + col.my * b_[1][1];
                const double g2 = col.mx * b_[0][2] + col.my * b_[1][2];
                const double c0 = b_[2][0], c1 = b_[2][1], c2 = b_[2][2];
#pragma omp simd
                for (int j = 0; j < jn; ++j) {
                    ax[j] += g0 * t1[j] + c0 * t2[j];
                    ay[j] += g1 * t1[j] + c1 * t2[j];
                    az[j] += g2 * t1[j] + c2 * t2[j];
                }
            }
            for (int j = 0; j < jn; ++j) {
                const double s = K2 * q[j0 + j];
                fx[j0 + j] += s * ax[j];
                fy[j0 + j] += s * ay[j];
                fz[j0 + j] += s * az[j];
            }
        }
    }
}

// T(k) = Σ_j q_j (k·u_j) e^{ik·r_j}; per column k·u = (m₀b₀ + m₁b₁)·u + m₂ b₂·u.
void EwaldSum::directional_factor(const EwaldPhases& ph, const StorageReal* q,
                                  const double* ux, const double* uy,
                                  const double* uz, int n, double* tr,
                                  double* ti) const {
    const int n_columns = static_cast<int>(columns_.size());
    const int team =
        team_for(static_cast<std::size_t>(n) * (kx_.size() + columns_.size()));
#pragma omp parallel for num_threads(team) if (team > 1) schedule(dynamic, 1)
    for (int c = 0; c < n_columns; ++c) {
        const Column col = columns_[c];
        for (int k = col.begin; k < col.end; ++k) {
            tr[k] = 0.0;
            ti[k] = 0.0;
        }
        alignas(64) double pr[kBlock], pi[kBlock];
        alignas(64) double ar[kBlock], ai[kBlock], br[kBlock], bi[kBlock];
        for (int j0 = 0; j0 < n; j0 += kBlock) {
            const int jn = std::min(kBlock, n - j0);
            column_phase(ph, col.mx, col.my, j0, jn, q, pr, pi);
            const double g0 = col.mx * b_[0][0] + col.my * b_[1][0];
            const double g1 = col.mx * b_[0][1] + col.my * b_[1][1];
            const double g2 = col.mx * b_[0][2] + col.my * b_[1][2];
#pragma omp simd
            for (int j = 0; j < jn; ++j) {
                const double u0 = ux[j0 + j], u1 = uy[j0 + j], u2 = uz[j0 + j];
                const double w = g0 * u0 + g1 * u1 + g2 * u2;
                const double v = b_[2][0] * u0 + b_[2][1] * u1 + b_[2][2] * u2;
                ar[j] = w * pr[j];
                ai[j] = w * pi[j];
                br[j] = v * pr[j];
                bi[j] = v * pi[j];
            }
            for (int k = col.begin; k < col.end; ++k) {
                const int mz = col.mz0 + (k - col.begin);
                const double m2 = mz;
                const double* __restrict cz = ph.cos_row(2, mz) + j0;
                const double* __restrict sz = ph.sin_row(2, mz) + j0;
                double re = 0.0, im = 0.0;
#pragma omp simd reduction(+ : re, im)
                for (int j = 0; j < jn; ++j) {
                    const double xr = ar[j] + m2 * br[j];
                    const double xi = ai[j] + m2 * bi[j];
                    re += xr * cz[j] - xi * sz[j];
                    im += xr * sz[j] + xi * cz[j];
                }
                tr[k] += re;
                ti[k] += im;
            }
        }
    }
}

// The header's three sums per atom j, with c + i s = e^{ik·r_j}: each sum
// over k of a term times k is split per column as
// (m₀b₀ + m₁b₁) Σ term + b₂ Σ m₂ term.
void EwaldSum::add_curvature(const EwaldPhases& ph, const StorageReal* q,
                             const double* ux, const double* uy,
                             const double* uz, int n, double volume,
                             const double* sr, const double* si,
                             const double* tr, const double* ti, double* hx,
                             double* hy, double* hz, double* lap, double* gx,
                             double* gy, double* gz) const {
    const double K = qqrd2e_ * 4.0 * ATV::units::pi / volume;
    const double* __restrict kx = kx_.data();
    const double* __restrict ky = ky_.data();
    const double* __restrict kz = kz_.data();
    const double* __restrict kw = kw_.data();
    const int team =
        team_for(static_cast<std::size_t>(n) * (kx_.size() + columns_.size()));
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int rank = 0, size = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num();
        size = omp_get_num_threads();
#endif
        int i0, i1;
        team_range(n, rank, size, i0, i1);
        alignas(64) double pr[kBlock], pi[kBlock], ku0[kBlock], ku2[kBlock];
        alignas(64) double h1[kBlock], h2[kBlock], g1[kBlock], g2[kBlock];
        alignas(64) double ah[3][kBlock], ag[3][kBlock], al[kBlock];
        for (int j0 = i0; j0 < i1; j0 += kBlock) {
            const int jn = std::min(kBlock, i1 - j0);
            for (int d = 0; d < 3; ++d) {
                std::fill_n(ah[d], jn, 0.0);
                std::fill_n(ag[d], jn, 0.0);
            }
            std::fill_n(al, jn, 0.0);
            for (int j = 0; j < jn; ++j)
                ku2[j] = b_[2][0] * ux[j0 + j] + b_[2][1] * uy[j0 + j]
                         + b_[2][2] * uz[j0 + j];
            for (const Column& col : columns_) {
                column_phase(ph, col.mx, col.my, j0, jn, nullptr, pr, pi);
                const double c0 = col.mx * b_[0][0] + col.my * b_[1][0];
                const double c1 = col.mx * b_[0][1] + col.my * b_[1][1];
                const double c2 = col.mx * b_[0][2] + col.my * b_[1][2];
                for (int j = 0; j < jn; ++j)
                    ku0[j] =
                        c0 * ux[j0 + j] + c1 * uy[j0 + j] + c2 * uz[j0 + j];
                std::fill_n(h1, jn, 0.0);
                std::fill_n(h2, jn, 0.0);
                std::fill_n(g1, jn, 0.0);
                std::fill_n(g2, jn, 0.0);
                for (int k = col.begin; k < col.end; ++k) {
                    const int mz = col.mz0 + (k - col.begin);
                    const double m2 = mz;
                    const double* __restrict cz = ph.cos_row(2, mz) + j0;
                    const double* __restrict sz = ph.sin_row(2, mz) + j0;
                    const double k2 =
                        kx[k] * kx[k] + ky[k] * ky[k] + kz[k] * kz[k];
                    const double w = kw[k], wk2 = kw[k] * k2;
                    const double s_r = sr[k], s_i = si[k];
                    const double t_r = tr[k], t_i = ti[k];
#pragma omp simd
                    for (int j = 0; j < jn; ++j) {
                        const double c = pr[j] * cz[j] - pi[j] * sz[j];
                        const double s = pr[j] * sz[j] + pi[j] * cz[j];
                        const double cs = c * s_r + s * s_i;
                        const double ku = ku0[j] + m2 * ku2[j];
                        const double hs = w * (c * t_r + s * t_i - ku * cs);
                        const double gs = wk2 * (s * s_r - c * s_i);
                        h1[j] += hs;
                        h2[j] += m2 * hs;
                        g1[j] += gs;
                        g2[j] += m2 * gs;
                        al[j] += wk2 * (q[j0 + j] - cs);
                    }
                }
                const double cc[3] = {c0, c1, c2};
                for (int d = 0; d < 3; ++d) {
                    const double e = b_[2][d];
#pragma omp simd
                    for (int j = 0; j < jn; ++j) {
                        ah[d][j] += cc[d] * h1[j] + e * h2[j];
                        ag[d][j] += cc[d] * g1[j] + e * g2[j];
                    }
                }
            }
            for (int j = 0; j < jn; ++j) {
                const double a = 2.0 * K * q[j0 + j];
                hx[j0 + j] += a * ah[0][j];
                hy[j0 + j] += a * ah[1][j];
                hz[j0 + j] += a * ah[2][j];
                lap[j0 + j] += a * al[j];
                gx[j0 + j] += 2.0 * a * ag[0][j];
                gy[j0 + j] += 2.0 * a * ag[1][j];
                gz[j0 + j] += 2.0 * a * ag[2][j];
            }
        }
    }
}

void EwaldSum::reciprocal_energy_virial(const double* sr, const double* si,
                                        double volume, double& e_recip,
                                        double stress[6]) const {
    const int ng = static_cast<int>(kx_.size());
    // Half-sphere prefactor 4π k_e / V (see rebuild() hemisphere filter).
    const double K = qqrd2e_ * 4.0 * ATV::units::pi / volume;
    const double* __restrict kx = kx_.data();
    const double* __restrict ky = ky_.data();
    const double* __restrict kz = kz_.data();
    const double* __restrict kw = kw_.data();
    const double* __restrict kvt = kvterm_.data();
    e_recip = 0;
    double s_xx = 0, s_yy = 0, s_zz = 0, s_xy = 0, s_xz = 0, s_yz = 0;
    for (int k = 0; k < ng; ++k) {
        const double sk2 = sr[k] * sr[k] + si[k] * si[k];
        const double ug = K * kw[k] * sk2;
        e_recip += ug;
        const double vt = kvt[k];
        s_xx += ug * (1.0 + vt * kx[k] * kx[k]);
        s_yy += ug * (1.0 + vt * ky[k] * ky[k]);
        s_zz += ug * (1.0 + vt * kz[k] * kz[k]);
        s_xy += ug * (vt * kx[k] * ky[k]);
        s_xz += ug * (vt * kx[k] * kz[k]);
        s_yz += ug * (vt * ky[k] * kz[k]);
    }
    stress[0] = s_xx;
    stress[1] = s_yy;
    stress[2] = s_zz;
    stress[3] = s_xy;
    stress[4] = s_xz;
    stress[5] = s_yz;
}

void EwaldSum::clear() noexcept {
    std::vector<double>().swap(kx_);
    std::vector<double>().swap(ky_);
    std::vector<double>().swap(kz_);
    std::vector<double>().swap(kw_);
    std::vector<double>().swap(kvterm_);
    std::vector<Column>().swap(columns_);
    mmax_ = {0, 0, 0};
}

std::size_t EwaldSum::memory_bytes() const noexcept {
    return (kx_.capacity() + ky_.capacity() + kz_.capacity() + kw_.capacity()
            + kvterm_.capacity())
               * sizeof(double)
           + columns_.capacity() * sizeof(Column);
}

} // namespace longrange
} // namespace ATV
