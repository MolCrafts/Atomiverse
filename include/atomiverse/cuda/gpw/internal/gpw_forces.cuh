/// @file cuda/gpw/internal/gpw_forces.cuh
/// @brief Device grid force product for CUDA GPW (Γ RKS first slice).
///
/// Gamma RKS single-grid cold geometry provides compact AO Pulay coefficients;
/// accepted device P and physical final-H W supply their device contraction.
/// Device Hellmann–Feynman / grid-Pulay terms consume resident V_eff / V_es:
///   * grid V_eff term −2 Σ_μ∈A Σ_g w V_eff χ_{dμ} (Pχ)_μ
///   * ionic HF +∫ V_es ∂n_c^A/∂R
///   * local-PP HF +∫ ρ ∂V_loc^SR/∂R  (when local PP is on the grid)
///   * Gaussian-core overlap pair forces, from immutable cold pair geometry
///
/// Not a public API — composed by `cuda::gpw::GpwDft::forces(Frame&)`.
#pragma once

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/electronic/gaussian_core.h"

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Max GTH local C_k coefficients packed for one atom (GTH ≤ 5).
constexpr int kMaxCloc = 8;

/// @brief One ion for device ionic / local-PP Hellmann–Feynman kernels.
struct DeviceForceAtom {
    double pos[3] = {0, 0, 0}; ///< nuclear position [Bohr].
    double z_val = 0.0;        ///< valence charge (ionic Gaussian).
    double rloc = 0.0;         ///< GTH local radius [Bohr].
    int n_cloc = 0;            ///< number of local SR coefficients.
    double cloc[kMaxCloc] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< C_k [Ha].
};

/// @brief Compact-basis CSR packed for device grid-force walks.
///
/// Layout matches `cpu::gpw::CompactBasisGrid`: per active grid point, a
/// contiguous (function index, value) list. `g2a` maps full-grid index →
/// cbg active index (−1 if inactive).
struct DeviceCbgView {
    const long* gidx = nullptr;       ///< active grid indices, length n_active.
    const std::size_t* off = nullptr; ///< CSR offsets, length n_active+1.
    const int* fu = nullptr;          ///< function indices, length nnz.
    const double* fv = nullptr;       ///< φ values, length nnz.
    const int* g2a = nullptr; ///< full-grid → cbg active (−1), length npts.
    int n_active = 0;
    long npts = 0;
};

/// @brief Grid V_eff Pulay forces on device (closed-shell RKS).
///
/// For each cbgd nonzero at active point with matching cbg density point:
///   F[owner[u], k] −= 2 w V_eff[g] χ_dμ (P χ)_μ
/// Caller zeros `d_forces` (length 3*n_atoms). Uses atomicAdd.
///
/// @param grid    Orthorhombic grid (host descriptor).
/// @param cbg     Undifferentiated χ CSR + g2a map.
/// @param cbgd    Center-derivative CSR (flat fu = u*3+k).
/// @param d_veff  Device V_eff, length npts [Hartree].
/// @param d_P     Device symmetric dense P, column-major, length nb*nb.
///                Symmetry makes the kernel's row indexing equivalent.
/// @param nb      AO basis size.
/// @param d_owner Device map AO index → atom index, length nb.
/// @param d_forces Device forces 3*n_atoms [Hartree/Bohr], accumulated.
/// @param stream  CUDA stream (null ⇒ default).
void accumulate_grid_veff_forces_device(
    const CollocateGridDesc& grid, const DeviceCbgView& cbg,
    const DeviceCbgView& cbgd, const double* d_veff, const double* d_P, int nb,
    const int* d_owner, double* d_forces, cudaStream_t stream = nullptr);

/// @brief Ionic core Hellmann–Feynman: F_A += w Σ_g V_es ∂n_c^A/∂R_A.
///
/// Min-image orthorhombic; n_c^A is the normalized Gaussian core of atom A.
///
/// @param grid    Orthorhombic grid.
/// @param d_atoms Device atoms, length n_atoms.
/// @param n_atoms Atom count.
/// @param d_v_es  Device electrostatic potential, length npts [Hartree].
/// @param periodic Min-image wrap (first-slice always true).
/// @param d_forces Device forces 3*n_atoms, accumulated.
/// @param stream  CUDA stream.
void accumulate_ionic_hf_forces_device(const CollocateGridDesc& grid,
                                       const DeviceForceAtom* d_atoms,
                                       int n_atoms, const double* d_v_es,
                                       bool periodic, double* d_forces,
                                       cudaStream_t stream = nullptr);

/// @brief Local GTH short-range PP Hellmann–Feynman on the grid.
///
/// Twin of `cpu::gpw::add_local_pp_hf_forces`. Skips atoms with n_cloc==0.
///
/// @param grid    Orthorhombic grid.
/// @param d_atoms Device atoms, length n_atoms.
/// @param n_atoms Atom count.
/// @param d_rho   Device electron density, length npts [e/Bohr³].
/// @param periodic Min-image wrap.
/// @param d_forces Device forces 3*n_atoms, accumulated.
/// @param stream  CUDA stream.
void accumulate_local_pp_hf_forces_device(const CollocateGridDesc& grid,
                                          const DeviceForceAtom* d_atoms,
                                          int n_atoms, const double* d_rho,
                                          bool periodic, double* d_forces,
                                          cudaStream_t stream = nullptr);

/// One exact-nonzero cold geometry contraction. Matrix index is column-major.
/// No atom*basis*basis dense tensor is allocated; screened zeros are omitted.
struct AoForceEntry {
    int force_lane = 0;
    int matrix_index = 0;
    double overlap = 0;  ///< coefficient multiplying physical W
    double one_body = 0; ///< coefficient multiplying accepted P
};

/// Cold geometry owns immutable sparse AO/grid tables and bounded force scratch.
/// evaluate() has device-only scientific inputs and performs no allocations or
/// host/device copies. Last-result/Frame retrieval is the method's outer boundary.
class GpwForceWorkspace {
  public:
    GpwForceWorkspace();
    ~GpwForceWorkspace();
    GpwForceWorkspace(const GpwForceWorkspace&) = delete;
    GpwForceWorkspace& operator=(const GpwForceWorkspace&) = delete;
    void build(const CollocateGridDesc& grid,
               const cpu::gpw::CompactBasisGrid& cbg,
               const cpu::gpw::CompactBasisGrid& cbgd,
               const std::vector<int>& owner,
               const std::vector<DeviceForceAtom>& atoms,
               const std::vector<AoForceEntry>& ao_entries,
               const std::vector<gaussian_core::Pair>& core_pairs,
               cudaStream_t stream);
    void evaluate(const double* d_P, const double* d_W, const double* d_veff,
                  const double* d_v_es, bool local_pp_on_grid);
    const double* d_forces() const;
    const double* d_rho() const;
    std::size_t geometry_bytes() const;
    std::size_t scratch_bytes() const;
    std::size_t ao_entries() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
