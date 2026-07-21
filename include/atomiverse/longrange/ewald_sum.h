/// @file longrange/ewald_sum.h
/// @brief Backend-agnostic reciprocal-space Ewald math (no Frame / Component /
///        MPI / FFT).
///
/// Pure operator math for the Ewald reciprocal sum (P. P. Ewald, *Ann. Phys.*
/// **369**, 253 (1921). DOI: 10.1002/andp.19213690304):
///
///   E_recip = units::coulomb · (2π/V) Σ_{k≠0} exp(−k²/4α²)/k² · |S(k)|²
///   E_self  = −units::coulomb · (α/√π) Σ_i q_i²
///   E_corr  = −units::coulomb · π/(2 V α²) · (Σ_i q_i)²
///
/// with S(k) = Σ_j q_j exp(i k·r_j). The MPI seam is intentionally between
/// #structure_factor (partial S(k) over owned atoms) and #add_forces /
/// #reciprocal_energy_virial (consume global S(k)) — Allreduce stays in the
/// backend wrapper.
///
/// Units: α [1/Å], k [Å⁻¹], r [Å], V [Å³], energy / virial [eV], force [eV/Å].
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace longrange {

/// @brief Reciprocal-space Ewald summation kernels over bare arrays.
///
/// Constructed with validated (α, accuracy); rebuilds its k-vector tables from
/// an inverse cell matrix. No Frame / Component / MPI / FFT.
class EwaldSum {
  public:
    /// @param alpha    Ewald splitting parameter α [1/Å]; must be > 0.
    /// @param accuracy Relative accuracy target in (0, 1); sets |k| cutoff.
    ///                 Default 1e-5.
    explicit EwaldSum(double alpha, double accuracy = 1e-5)
        : alpha_(alpha), accuracy_(accuracy) {
        if (alpha <= 0.0)
            throw std::invalid_argument("EwaldSum: alpha must be > 0");
        if (!(accuracy > 0.0 && accuracy < 1.0))
            throw std::invalid_argument("EwaldSum: accuracy must be in (0, 1)");
    }

    /// @brief Recommended α [1/Å] for a real-space cutoff and accuracy
    ///        (LAMMPS heuristic g_ewald = (1.35 − 0.15·ln(accuracy)) / cutoff).
    /// @param cutoff   Real-space cutoff [Å].
    /// @param accuracy Relative accuracy target in (0, 1). Default 1e-5.
    /// @return α [1/Å].
    static double recommend_alpha(double cutoff, double accuracy = 1e-5) {
        return (1.35 - 0.15 * std::log(accuracy)) / cutoff;
    }

    /// @brief Rebuild reciprocal-vector tables from the inverse cell.
    /// @param inv_cell Inverse of the real-space cell matrix [1/Å]
    ///                 (Mat3 indexing: inv_cell(row, col)).
    /// @return true if any internal table capacity grew (reserve).
    bool rebuild(const Mat3& inv_cell) {
        // Reciprocal basis b_i = 2π · (column i of cell⁻¹); g = Σ m_i b_i.
        const double tp = 2.0 * ATV::units::pi;
        double b[3][3];
        for (int i = 0; i < 3; ++i) {
            b[i][0] = tp * inv_cell(0, i);
            b[i][1] = tp * inv_cell(1, i);
            b[i][2] = tp * inv_cell(2, i);
        }
        auto blen = [&](int i) {
            return std::sqrt(b[i][0] * b[i][0] + b[i][1] * b[i][1]
                             + b[i][2] * b[i][2]);
        };
        const double gcut = 2.0 * alpha_ * std::sqrt(-std::log(accuracy_));
        const double gcut2 = gcut * gcut;
        int kmax[3];
        for (int i = 0; i < 3; ++i)
            kmax[i] = std::max(1, static_cast<int>(std::ceil(gcut / blen(i))));

        const std::size_t max_vectors =
            static_cast<std::size_t>(2 * kmax[0] + 1)
                * static_cast<std::size_t>(2 * kmax[1] + 1)
                * static_cast<std::size_t>(2 * kmax[2] + 1)
            - 1;
        const bool capacity_grows =
            kx_.capacity() < max_vectors || ky_.capacity() < max_vectors
            || kz_.capacity() < max_vectors || kw_.capacity() < max_vectors
            || kvterm_.capacity() < max_vectors;
        if (capacity_grows) {
            kx_.reserve(max_vectors);
            ky_.reserve(max_vectors);
            kz_.reserve(max_vectors);
            kw_.reserve(max_vectors);
            kvterm_.reserve(max_vectors);
        }

        const double inv4a2 = 1.0 / (4.0 * alpha_ * alpha_);
        kx_.clear();
        ky_.clear();
        kz_.clear();
        kw_.clear();
        kvterm_.clear();
        for (int mx = -kmax[0]; mx <= kmax[0]; ++mx)
            for (int my = -kmax[1]; my <= kmax[1]; ++my)
                for (int mz = -kmax[2]; mz <= kmax[2]; ++mz) {
                    if (mx == 0 && my == 0 && mz == 0)
                        continue;
                    const double gx =
                        mx * b[0][0] + my * b[1][0] + mz * b[2][0];
                    const double gy =
                        mx * b[0][1] + my * b[1][1] + mz * b[2][1];
                    const double gz =
                        mx * b[0][2] + my * b[1][2] + mz * b[2][2];
                    const double g2 = gx * gx + gy * gy + gz * gz;
                    if (g2 > gcut2)
                        continue;
                    kx_.push_back(gx);
                    ky_.push_back(gy);
                    kz_.push_back(gz);
                    kw_.push_back(std::exp(-g2 * inv4a2) / g2);
                    kvterm_.push_back(-2.0 * (1.0 / g2 + inv4a2));
                }
        return capacity_grows;
    }

    /// @return Number of retained reciprocal vectors.
    std::size_t size() const noexcept { return kx_.size(); }

    /// @return Reciprocal-vector k_x entries [Å⁻¹], length #size().
    const double* kx() const noexcept { return kx_.data(); }
    /// @return Reciprocal-vector k_y entries [Å⁻¹].
    const double* ky() const noexcept { return ky_.data(); }
    /// @return Reciprocal-vector k_z entries [Å⁻¹].
    const double* kz() const noexcept { return kz_.data(); }
    /// @return Kernel weights A(k) = exp(−k²/4α²)/k² [Å²].
    const double* kw() const noexcept { return kw_.data(); }
    /// @return Per-k virial geometric factor vterm = −2(1/k² + 1/4α²) [Å²].
    const double* kvterm() const noexcept { return kvterm_.data(); }

    /// @brief Partial structure factor S(k) over @p n atoms into @p sr / @p si.
    /// @param x,y,z Atom coordinates [Å], length @p n.
    /// @param q     Charges [e], length @p n.
    /// @param n     Atom count.
    /// @param[out] sr Real part of S(k), length #size().
    /// @param[out] si Imag part of S(k), length #size().
    void structure_factor(const StorageReal* x, const StorageReal* y,
                          const StorageReal* z, const StorageReal* q, int n,
                          double* sr, double* si) const {
        const int ng = static_cast<int>(kx_.size());
        const double* __restrict kx = kx_.data();
        const double* __restrict ky = ky_.data();
        const double* __restrict kz = kz_.data();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < ng; ++k) {
            double sre = 0, sim = 0;
            const double kxk = kx[k], kyk = ky[k], kzk = kz[k];
            for (int i = 0; i < n; ++i) {
                const double phase = kxk * x[i] + kyk * y[i] + kzk * z[i];
                sre += q[i] * std::cos(phase);
                sim += q[i] * std::sin(phase);
            }
            sr[k] = sre;
            si[k] = sim;
        }
    }

    /// @brief Add reciprocal forces on @p n atoms from global S(k).
    /// @param x,y,z Atom coordinates [Å], length @p n.
    /// @param q     Charges [e], length @p n.
    /// @param n     Atom count.
    /// @param volume Cell volume [Å³].
    /// @param sr,si Global structure factor, length #size().
    /// @param[in,out] fx,fy,fz Force accumulators [eV/Å], length @p n.
    void add_forces(const StorageReal* x, const StorageReal* y,
                    const StorageReal* z, const StorageReal* q, int n,
                    double volume, const double* sr, const double* si,
                    AccReal* fx, AccReal* fy, AccReal* fz) const {
        const int ng = static_cast<int>(kx_.size());
        const double K = ATV::units::coulomb * 2.0 * ATV::units::pi / volume;
        const double* __restrict kx = kx_.data();
        const double* __restrict ky = ky_.data();
        const double* __restrict kz = kz_.data();
        const double* __restrict kw = kw_.data();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double xi = x[i], yi = y[i], zi = z[i], qi = q[i];
            if (qi == 0.0)
                continue;
            double fxi = 0, fyi = 0, fzi = 0;
            for (int k = 0; k < ng; ++k) {
                const double phase = kx[k] * xi + ky[k] * yi + kz[k] * zi;
                const double c = std::cos(phase), s = std::sin(phase);
                const double fac =
                    2.0 * K * kw[k] * qi * (s * sr[k] - c * si[k]);
                fxi += fac * kx[k];
                fyi += fac * ky[k];
                fzi += fac * kz[k];
            }
            fx[i] += fxi;
            fy[i] += fyi;
            fz[i] += fzi;
        }
    }

    /// @brief Self-energy correction [eV] for Σ q_i² over owned atoms.
    /// @param qsq_sum Σ_i q_i² [e²].
    /// @return E_self [eV].
    double self_energy(double qsq_sum) const {
        return -ATV::units::coulomb * alpha_ / std::sqrt(ATV::units::pi)
               * qsq_sum;
    }

    /// @brief Net-charge (background) correction [eV].
    /// @param q_total Total charge Σ_i q_i [e].
    /// @param volume  Cell volume [Å³].
    /// @return E_corr [eV].
    double net_charge_energy(double q_total, double volume) const {
        return -ATV::units::coulomb * ATV::units::pi
               / (2.0 * volume * alpha_ * alpha_) * q_total * q_total;
    }

    /// @brief Reciprocal energy and virial from global S(k).
    /// @param sr,si  Global structure factor, length #size().
    /// @param volume Cell volume [Å³].
    /// @param[out] e_recip Reciprocal energy [eV].
    /// @param[out] stress  Virial in Voigt order xx,yy,zz,xy,xz,yz [eV].
    void reciprocal_energy_virial(const double* sr, const double* si,
                                  double volume, double& e_recip,
                                  double stress[6]) const {
        const int ng = static_cast<int>(kx_.size());
        const double K = ATV::units::coulomb * 2.0 * ATV::units::pi / volume;
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

    /// @return α [1/Å].
    double alpha() const noexcept { return alpha_; }
    /// @return Relative accuracy target.
    double accuracy() const noexcept { return accuracy_; }

    /// @brief Drop k-vector storage.
    void clear() noexcept {
        std::vector<double>().swap(kx_);
        std::vector<double>().swap(ky_);
        std::vector<double>().swap(kz_);
        std::vector<double>().swap(kw_);
        std::vector<double>().swap(kvterm_);
    }

    /// @return Bytes of reserved k-vector storage.
    std::size_t memory_bytes() const noexcept {
        return (kx_.capacity() + ky_.capacity() + kz_.capacity()
                + kw_.capacity() + kvterm_.capacity())
               * sizeof(double);
    }

  private:
    double alpha_;
    double accuracy_;
    std::vector<double> kx_, ky_, kz_, kw_, kvterm_;
};

} // namespace longrange
} // namespace ATV
