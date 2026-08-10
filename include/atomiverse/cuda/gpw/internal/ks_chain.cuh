/// @file cuda/gpw/internal/ks_chain.cuh
/// @brief Composed Γ-point device KS stage: products → ρ → Poisson → XC → V_eff → integrate.
///
/// Stage unit reusing landed footholds (`product_collocate`, `PoissonFftDevice`,
/// `xc_lda_grid`/`xc_energy`, `product_integrate`). Mirrors the fine-grid path
/// of CPU `ks_grid_build_rks` (closed-shell Padé LDA, periodic Poisson) plus
/// collocate/integrate bookends. **Not** a full GPW method product: no
/// multigrid, no k-points, no GGA/vdW, no public `GpwDft` surface.
#pragma once

#include "atomiverse/cuda/gpw/internal/poisson_fft.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/product_integrate.cuh"
#include "atomiverse/cuda/gpw/internal/xc_lda.cuh"

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Host-side scalars from one KS-chain launch [Hartree].
struct KsChainEnergies {
    double e_hartree = 0.0; ///< 0.5 w Σ n V_es (Poisson of n = ρ − n_ion).
    double e_xc = 0.0;      ///< w Σ ρ ε_xc (Padé LDA).
    double e_locgrid = 0.0; ///< w Σ ρ V_loc when V_loc provided; else 0.
};

/// @brief Device-resident closed-shell KS chain for one orthorhombic grid shape.
///
/// Owns workspace + a `PoissonFftDevice`. Caller owns product arrays and
/// optional ion / local-PP grids. All scientific work stays on device after
/// products (and optional n_ion / V_loc) are already on device.
class KsChainDevice {
    CollocateGridDesc coll_grid_{};
    GpwGridDesc poisson_grid_{};
    long ntot_ = 0;
    double weight_ = 0.0;
    cudaStream_t stream_ = nullptr;

    PoissonFftDevice poisson_;

    double* d_rho_ = nullptr;    ///< electron density ρ
    double* d_n_tot_ = nullptr;  ///< n = ρ − n_ion (Poisson input)
    double* d_v_es_ = nullptr;   ///< electrostatic potential
    double* d_vxc_ = nullptr;    ///< XC potential
    double* d_exc_ = nullptr;    ///< per-electron ε_xc
    double* d_veff_ = nullptr;   ///< V_es + V_xc [+ V_loc]
    double* d_e_xc_ = nullptr;   ///< scalar E_xc
    double* d_e_loc_ = nullptr;  ///< scalar ∫ρ V_loc / then ×w

  public:
    /// @brief Allocate workspace + Poisson plans for grid shape / cell.
    /// @param coll   Orthorhombic collocate grid (n, L, origin) [Bohr].
    /// @param stream CUDA stream (default stream if null).
    explicit KsChainDevice(const CollocateGridDesc& coll,
                           cudaStream_t stream = nullptr);
    ~KsChainDevice();

    KsChainDevice(const KsChainDevice&) = delete;
    KsChainDevice& operator=(const KsChainDevice&) = delete;

    /// @brief Run the full chain on device.
    ///
    /// Steps (all device):
    ///  1. zero ρ; collocate products → ρ
    ///  2. n = ρ − n_ion (n_ion null ⇒ n = ρ)
    ///  3. Poisson(n) → V_es, e_hartree
    ///  4. Padé LDA(ρ) → V_xc, ε_xc; E_xc = w Σ ρ ε_xc
    ///  5. V_eff = V_es + V_xc [+ V_loc]; optional e_locgrid = w Σ ρ V_loc
    ///  6. if d_acc: integrate V_eff against product shapes → d_acc[0..nprod)
    ///
    /// @param d_prods Device products, length nprod (collocate uses weight;
    ///                integrate is shape-only).
    /// @param nprod   Number of products (≥ 1 for a meaningful density).
    /// @param d_n_ion Optional ionic Gaussian density, length npts (null ⇒ 0).
    /// @param d_v_loc Optional short-range local PP, length npts (null ⇒ 0).
    /// @param d_acc   Optional per-product integrate output, length nprod
    ///                (null ⇒ skip integrate).
    /// @return Host-side energy pieces after device sync of scalars.
    KsChainEnergies run(const DeviceProduct* d_prods, int nprod,
                        const double* d_n_ion = nullptr,
                        const double* d_v_loc = nullptr,
                        double* d_acc = nullptr);

    /// @brief Device pointer to last V_eff (length npts) [Hartree].
    const double* d_veff() const { return d_veff_; }
    /// @brief Device pointer to last ρ (length npts).
    const double* d_rho() const { return d_rho_; }
    /// @brief Device pointer to last V_es (length npts).
    const double* d_v_es() const { return d_v_es_; }
    /// @brief Device pointer to last V_xc (length npts).
    const double* d_vxc() const { return d_vxc_; }

    long ntot() const { return ntot_; }
    double weight() const { return weight_; }
    const CollocateGridDesc& coll_grid() const { return coll_grid_; }
    cudaStream_t stream() const { return stream_; }
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
