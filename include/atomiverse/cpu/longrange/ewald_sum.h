/// @file cpu/longrange/ewald_sum.h
/// @brief Backend-agnostic reciprocal-space Ewald math (no Frame / Component /
///        MPI / FFT).
///
/// Pure operator math for the Ewald reciprocal sum (P. P. Ewald, *Ann. Phys.*
/// **369**, 253 (1921). DOI: 10.1002/andp.19213690304):
///
///   E_recip = qqrd2e · (4π/V) Σ_{k∈½} exp(−k²/4α²)/k² · |S(k)|²
///   E_self  = −qqrd2e · (α/√π) Σ_i q_i²
///   E_corr  = −qqrd2e · π/(2 V α²) · (Σ_i q_i)²
///
/// `qqrd2e` is the run's Coulomb constant (`set_qqrd2e`, LAMMPS
/// `force->qqrd2e`; default 1, the lj row).
///
/// with S(k) = Σ_j q_j exp(i k·r_j). The k-sum is over the **positive
/// hemisphere** only (kz>0, or kz=0∧ky>0, or kz=ky=0∧kx>0), matching the
/// LAMMPS Ewald half-sphere convention with prefactor 4π/V (equivalent to a
/// full-sphere sum at 2π/V). The MPI seam is intentionally between
/// #structure_factor (partial S(k) over owned atoms) and #add_forces /
/// #reciprocal_energy_virial (consume global S(k)) — Allreduce stays in the
/// backend wrapper.
///
/// **Phase factors.** Every k is an integer combination
/// k = m₀b₀ + m₁b₁ + m₂b₂ of the reciprocal basis, so
/// exp(ik·r) = Π_a exp(i m_a b_a·r). #phases evaluates exp(i b_a·r_j) once
/// per atom and axis and builds the powers by complex recurrence (LAMMPS
/// `Ewald::eik_dot_r`); the kernels then take products of table entries
/// instead of a sine and a cosine per (k, atom) pair. The vectors are stored
/// in columns of fixed (m₀, m₁) with consecutive m₂, so the in-plane factor is
/// formed once per column.
///
/// Units: α [1/Å], k [Å⁻¹], r [Å], V [Å³], energy / virial [eV], force [eV/Å].
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace longrange {

/// @brief Per-atom powers exp(i m b_a·r_j) of the three reciprocal basis
///        vectors, m = 0..`mmax[a]` (`EwaldSum::phases` fills them). A
///        negative m is the complex conjugate.
struct EwaldPhases {
    int n = 0;                                 ///< atoms
    std::array<int, 3> mmax{0, 0, 0};          ///< highest power per axis
    std::array<std::vector<double>, 3> re, im; ///< [m · n + j]

    /// @return Re exp(i m b_a·r_j) for j = 0..n−1 (m ≥ 0).
    const double* cos_row(int a, int m) const {
        return re[a].data() + static_cast<std::size_t>(m) * n;
    }
    /// @return Im exp(i m b_a·r_j) for j = 0..n−1 (m ≥ 0).
    const double* sin_row(int a, int m) const {
        return im[a].data() + static_cast<std::size_t>(m) * n;
    }
    /// @return Bytes of reserved table storage.
    std::size_t memory_bytes() const noexcept {
        std::size_t doubles = 0;
        for (int a = 0; a < 3; ++a)
            doubles += re[a].capacity() + im[a].capacity();
        return doubles * sizeof(double);
    }
};

/// @brief Reciprocal-space Ewald summation kernels over bare arrays.
///
/// Constructed with validated (α, accuracy); rebuilds its k-vector tables from
/// an inverse cell matrix. No Frame / Component / MPI / FFT.
///
/// The kernels open an OpenMP team of at most `omp_get_max_threads()`
/// threads, sized so that every thread gets at least
/// #kPairsPerThread (k-vector, atom) pairs: a small cell runs serially
/// instead of paying a team's fork and join for microseconds of work. Each
/// S(k) and each atom's force is summed in the same order whatever the team,
/// so the results do not depend on the thread count.
class EwaldSum {
  public:
    /// @brief (k-vector, atom) pairs one more thread must get to be worth
    ///        waking: ≈ 0.1 ms of vectorized work at ≈ 0.2 ns per pair,
    ///        several times an unbound team's fork, barrier and join on a
    ///        shared 2-socket node (measured on EPYC 9755).
    static constexpr std::size_t kPairsPerThread = std::size_t{1} << 19;

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
    ///                 (Mat3 indexing: inv_cell(row, col); real cell columns are lattice vectors).
    /// @return true if any internal table capacity grew (reserve).
    bool rebuild(const Mat3& inv_cell);

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

    /// @brief Phase tables of @p n atoms for the current reciprocal basis.
    /// @param x,y,z Atom coordinates [Å], length @p n.
    /// @param n     Atom count.
    /// @param[out] out Tables, resized to the basis' highest powers.
    void phases(const StorageReal* x, const StorageReal* y,
                const StorageReal* z, int n, EwaldPhases& out) const;

    /// @brief Partial structure factor S(k) over @p n atoms into @p sr / @p si.
    /// @param ph    Phase tables of the atoms (#phases).
    /// @param q     Charges [e], length @p n.
    /// @param n     Atom count.
    /// @param[out] sr Real part of S(k), length #size().
    /// @param[out] si Imag part of S(k), length #size().
    void structure_factor(const EwaldPhases& ph, const StorageReal* q, int n,
                          double* sr, double* si) const;

    /// @brief Add reciprocal forces on @p n atoms from global S(k).
    /// @param ph    Phase tables of the atoms (#phases).
    /// @param q     Charges [e], length @p n.
    /// @param n     Atom count.
    /// @param volume Cell volume [Å³].
    /// @param sr,si Global structure factor, length #size().
    /// @param[in,out] fx,fy,fz Force accumulators [eV/Å], length @p n.
    void add_forces(const EwaldPhases& ph, const StorageReal* q, int n,
                    double volume, const double* sr, const double* si,
                    AccReal* fx, AccReal* fy, AccReal* fz) const;

    /// @brief Directional structure factor
    ///        \(T(k) = \sum_j q_j (k\cdot u_j)\, e^{i k\cdot r_j}\) of a
    ///        per-atom vector field \(u\) (the Hessian product needs it).
    /// @param x,y,z    Atom coordinates [Å], length @p n.
    /// @param q        Charges [e], length @p n.
    /// @param ux,uy,uz Vector field, length @p n.
    /// @param n        Atom count.
    /// @param[out] tr,ti Real / imaginary part of T(k), length #size().
    void directional_factor(const StorageReal* x, const StorageReal* y,
                            const StorageReal* z, const StorageReal* q,
                            const double* ux, const double* uy,
                            const double* uz, int n, double* tr,
                            double* ti) const;

    /// @brief Second derivatives of the reciprocal energy
    ///        \(E = K\sum_{k\in\frac12} A_k |S_k|^2\),
    ///        \(K = 4\pi k_e/V\), added to @p n atoms.
    ///
    /// With \(c_i = \cos k\cdot r_i\), \(s_i = \sin k\cdot r_i\):
    ///  - \((Hu)_i = 2K\sum A\,q_i\,k\,[c_iT_r + s_iT_i
    ///    - (k\cdot u_i)(c_iS_r + s_iS_i)]\);
    ///  - \(\nabla_i^2 E = 2K\sum A\,k^2 q_i\,[q_i - (c_iS_r + s_iS_i)]\);
    ///  - \(\nabla_i\sum_j\nabla_j^2 E
    ///    = 4K\sum A\,k^2 q_i\,k\,(s_iS_r - c_iS_i)\)
    ///    (from \(\sum_j\nabla_j^2 E = 2K\sum A k^2(\sum q^2 - |S|^2)\)).
    /// @param sr,si Global S(k); @p tr,ti global T(k) of the same @p u.
    void add_curvature(const StorageReal* x, const StorageReal* y,
                       const StorageReal* z, const StorageReal* q,
                       const double* ux, const double* uy, const double* uz,
                       int n, double volume, const double* sr, const double* si,
                       const double* tr, const double* ti, double* hx,
                       double* hy, double* hz, double* lap, double* gx,
                       double* gy, double* gz) const;

    /// @brief Self-energy correction [eV] for Σ q_i² over owned atoms.
    /// @param qsq_sum Σ_i q_i² [e²].
    /// @return E_self [eV].
    double self_energy(double qsq_sum) const {
        return longrange::self_energy_prefactor(alpha_, qqrd2e_) * qsq_sum;
    }

    /// @brief Net-charge (background) correction [eV].
    /// @param q_total Total charge Σ_i q_i [e].
    /// @param volume  Cell volume [Å³].
    /// @return E_corr [eV].
    double net_charge_energy(double q_total, double volume) const {
        return longrange::background_energy(q_total, volume, alpha_, qqrd2e_);
    }

    /// @brief Reciprocal energy and virial from global S(k).
    /// @param sr,si  Global structure factor, length #size().
    /// @param volume Cell volume [Å³].
    /// @param[out] e_recip Reciprocal energy [eV].
    /// @param[out] stress  Virial in Voigt order xx,yy,zz,xy,xz,yz [eV].
    void reciprocal_energy_virial(const double* sr, const double* si,
                                  double volume, double& e_recip,
                                  double stress[6]) const;

    /// @return α [1/Å].
    double alpha() const noexcept { return alpha_; }
    /// @return Relative accuracy target.
    double accuracy() const noexcept { return accuracy_; }

    /// @brief Drop k-vector storage.
    void clear() noexcept;

    /// @return Bytes of reserved k-vector storage.
    std::size_t memory_bytes() const noexcept;

    /// @brief The run's Coulomb constant (LAMMPS `force->qqrd2e`); the
    ///        owning solver sets it at build from `ctx.runtime.units`.
    void set_qqrd2e(double qqrd2e) { qqrd2e_ = qqrd2e; }

  private:
    /// Reciprocal vectors k = m₀b₀ + m₁b₁ + m₂b₂ with m₂ = `mz0`, `mz0` + 1,
    /// … for k in [`begin`, `end`) of the tables.
    struct Column {
        int mx, my, mz0;
        int begin, end;
    };

    double qqrd2e_ = 1.0;
    double alpha_;
    double accuracy_;
    std::vector<double> kx_, ky_, kz_, kw_, kvterm_;
    std::vector<Column> columns_;
    double b_[3][3] = {}; ///< reciprocal basis, b_[a] = 2π · row a of cell⁻¹
    std::array<int, 3> mmax_{0, 0, 0}; ///< highest |m_a| of a kept vector
};

} // namespace longrange
} // namespace ATV
