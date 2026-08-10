/// @file cuda/gpw/internal/ks_chain.cuh
/// @brief Composed Γ-point device KS stage: products → ρ → Poisson → XC → V_eff → integrate.
///
/// Stage unit reusing landed footholds (`product_collocate`, `PoissonFftDevice`,
/// Padé LDA / closed-shell PBE via spectral ∇ρ/−∇·flux + `xc_pbe_grid`,
/// `product_integrate`, spectral multigrid prolong/restrict). Mirrors the
/// fine-grid path of CPU `ks_grid_build_rks` (closed-shell Padé LDA or PBE-GGA,
/// periodic Poisson) plus collocate/integrate bookends, and the multi-level
/// product density assembly of CPU Γ multigrid (collocate per level → spectral
/// prolong to fine → KS on fine → spectral restrict → integrate per level).
/// **Not** a full GPW method product: no k-points, no spin/vdW, no public
/// `GpwDft` surface.
#pragma once

#include "atomiverse/cuda/gpw/internal/multigrid_transfer.cuh"
#include "atomiverse/cuda/gpw/internal/poisson_fft.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/product_integrate.cuh"
#include "atomiverse/cuda/gpw/internal/spectral_fft.cuh"
#include "atomiverse/cuda/gpw/internal/xc_lda.cuh"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief XC functional for one KS-chain launch.
///
/// `PADE` = GTH Padé LDA (default; original chain). `PBE` = closed-shell PBE
/// GGA (spectral ∇ρ, device PBE-X/C, spectral −∇·flux). Requires the plane-wave
/// CUDA product (`ATV_HAS_PW`) for the shared `xc_pbe_grid` kernel.
enum class KsXcKind {
    PADE = 0,
    PBE = 1,
};

/// @brief Host-side scalars from one KS-chain launch [Hartree].
struct KsChainEnergies {
    double e_hartree = 0.0; ///< 0.5 w Σ n V_es (Poisson of n = ρ − n_ion).
    double e_xc = 0.0;      ///< w Σ ρ ε_xc (Padé LDA or PBE).
    double e_locgrid = 0.0; ///< w Σ ρ V_loc when V_loc provided; else 0.
};

/// @brief Device-resident closed-shell KS chain for one orthorhombic grid shape.
///
/// Owns workspace + a `PoissonFftDevice` + a `SpectralFftDevice` (GGA). Caller
/// owns product arrays and optional ion / local-PP grids. All scientific work
/// stays on device after products (and optional n_ion / V_loc) are already on
/// device.
class KsChainDevice {
    CollocateGridDesc coll_grid_{};
    GpwGridDesc poisson_grid_{};
    long ntot_ = 0;
    double weight_ = 0.0;
    cudaStream_t stream_ = nullptr;

    PoissonFftDevice poisson_;
    SpectralFftDevice spectral_;

    double* d_rho_ = nullptr;    ///< electron density ρ
    double* d_n_tot_ = nullptr;  ///< n = ρ − n_ion (Poisson input)
    double* d_v_es_ = nullptr;   ///< electrostatic potential
    double* d_vxc_ = nullptr;    ///< XC potential (local part, then full)
    double* d_exc_ = nullptr;    ///< per-electron ε_xc
    double* d_veff_ = nullptr;   ///< V_es + V_xc [+ V_loc]
    double* d_e_xc_ = nullptr;   ///< scalar E_xc
    double* d_e_loc_ = nullptr;  ///< scalar ∫ρ V_loc / then ×w
    // GGA scratch: ∇ρ / flux components + divergence of flux.
    double* d_gx_ = nullptr;
    double* d_gy_ = nullptr;
    double* d_gz_ = nullptr;
    double* d_div_ = nullptr;

  public:
    /// @brief Allocate workspace + Poisson / spectral plans for grid shape / cell.
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
    ///  4. XC(ρ) → V_xc, ε_xc; E_xc = w Σ ρ ε_xc
    ///     - PADE: Padé LDA on total ρ
    ///     - PBE:  spectral ∇ρ → closed-shell PBE (local v_ρ, flux) →
    ///             spectral −∇·flux → V_xc = v_ρ − div
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
    /// @param xc      XC functional (`PADE` default; `PBE` needs `ATV_HAS_PW`).
    /// @return Host-side energy pieces after device sync of scalars.
    KsChainEnergies run(const DeviceProduct* d_prods, int nprod,
                        const double* d_n_ion = nullptr,
                        const double* d_v_loc = nullptr,
                        double* d_acc = nullptr,
                        KsXcKind xc = KsXcKind::PADE);

    /// @brief Run Poisson+XC+assemble from a pre-built fine-grid density.
    ///
    /// Skips product collocate. Copies `d_rho_in` into internal ρ, then steps
    /// 2–5 of `run`. Optional integrate against `d_prods` on this (fine) grid
    /// only — multigrid integrate-on-coarse is `KsMultiGridChainDevice`.
    ///
    /// @param d_rho_in Device density, length npts (not modified).
    /// @param d_n_ion  Optional ionic density on the same grid.
    /// @param d_v_loc  Optional local PP on the same grid.
    /// @param d_prods  Optional products for fine-grid integrate (null + nprod
    ///                 0 ⇒ skip integrate).
    /// @param nprod    Number of products for optional integrate.
    /// @param d_acc    Optional integrate output (null ⇒ skip).
    /// @param xc       XC functional.
    KsChainEnergies run_from_rho(const double* d_rho_in,
                                 const double* d_n_ion = nullptr,
                                 const double* d_v_loc = nullptr,
                                 const DeviceProduct* d_prods = nullptr,
                                 int nprod = 0, double* d_acc = nullptr,
                                 KsXcKind xc = KsXcKind::PADE);

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

/// @brief Multi-grid product collocate → fine KS → restrict → integrate.
///
/// Mirrors the CPU Γ multigrid fragment:
///   for l: ρ_l = collocate(prods_l on level l)
///   ρ_fine = ρ_0 + Σ_{l>0} prolong_fft(ρ_l)
///   (V_eff, energies) = KS(ρ_fine) on finest grid
///   for l: V_l = (l==0 ? V_eff : restrict_fft(V_eff)); integrate prods_l vs V_l
///
/// Stage unit only — continuous product shapes (same footholds as single-grid
/// chain), not full collint screening / P-scatter / public `GpwDft`.
class KsMultiGridChainDevice {
    MultigridDesc mg_{};
    cudaStream_t stream_ = nullptr;
    int nlev_ = 0;

    std::unique_ptr<KsChainDevice> fine_;
    /// Transfer engine for level l>0 → fine (index l-1).
    std::vector<std::unique_ptr<MultigridTransferDevice>> xfer_;

    std::vector<double*> d_rho_level_; ///< per-level collocate buffers
    std::vector<double*> d_v_level_;   ///< restricted V for l>0 (l=0 uses fine)
    double* d_rho_fine_ = nullptr;     ///< accumulated fine density
    double* d_prolong_scratch_ = nullptr; ///< one fine-sized prolong buffer

  public:
    /// @brief Allocate fine KS chain + per-level buffers + spectral transfers.
    /// @param mg Multigrid (finest first; ≥1 level; even n; shared L).
    /// @param stream CUDA stream (default if null).
    explicit KsMultiGridChainDevice(const MultigridDesc& mg,
                                    cudaStream_t stream = nullptr);
    ~KsMultiGridChainDevice();

    KsMultiGridChainDevice(const KsMultiGridChainDevice&) = delete;
    KsMultiGridChainDevice& operator=(const KsMultiGridChainDevice&) = delete;

    /// @brief Multi-grid chain launch.
    ///
    /// @param d_prods_by_level Host array of device product pointers, length
    ///        nlevels. Level l may be null only if nprod_by_level[l]==0.
    /// @param nprod_by_level   Host array of product counts, length nlevels.
    ///        At least one level must have nprod > 0.
    /// @param d_n_ion Optional ionic density on the **fine** grid.
    /// @param d_v_loc Optional local PP on the **fine** grid.
    /// @param d_acc_by_level Optional host array of device acc pointers (length
    ///        nlevels). Null array ⇒ skip all integrate. Per-level null ⇒ skip
    ///        that level. Each non-null buffer length = nprod_by_level[l].
    /// @param xc XC on the fine grid.
    KsChainEnergies run(const DeviceProduct* const* d_prods_by_level,
                        const int* nprod_by_level,
                        const double* d_n_ion = nullptr,
                        const double* d_v_loc = nullptr,
                        double* const* d_acc_by_level = nullptr,
                        KsXcKind xc = KsXcKind::PADE);

    const KsChainDevice& fine_chain() const { return *fine_; }
    const MultigridDesc& multigrid() const { return mg_; }
    int nlevels() const { return nlev_; }
    cudaStream_t stream() const { return stream_; }
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
