/// @file cuda/gpw/internal/product_weight_pack.cuh
/// @brief Device P→product weight + acc→V_AO scatter for GPW mid-SCF.
///
/// CUDA twin of host `product_density_weight` / `scatter_product_acc` in
/// `gpw_solver.cu`. Geometry-fixed product descriptors and prim/contrib
/// tables live on device for the run; each SCF iteration only refreshes
/// `DeviceProduct.weight` from a resident density `d_P` and scatters the
/// integrate accumulator into the AO potential matrix.
///
/// Domain-local to `cuda::gpw::internal` — not a public method API.
#pragma once

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device slice of `cpu::gpw::PrimRef` needed for weight/scatter.
struct DevicePrimRef {
    int cbeg = 0; ///< start index into contrib pool.
    int cn = 0;   ///< contributor count.
};

/// @brief Device twin of `cpu::gpw::PrimContrib`.
struct DevicePrimContrib {
    int mu = 0;       ///< AO index.
    double coeff = 0; ///< harmonic·contraction coefficient.
};

/// @brief Geometry-fixed product meta (collint shape + P-weight indices).
///
/// `ua`/`ub`/`k` enter the density weight; remaining fields mirror
/// `DeviceProduct` geometry so pack can write a full product descriptor.
struct DeviceProductGeom {
    double p = 0.0;
    double pc[3] = {0, 0, 0};
    double ra[3] = {0, 0, 0};
    double rb[3] = {0, 0, 0};
    int la[3] = {0, 0, 0};
    int lb[3] = {0, 0, 0};
    int ua = 0;   ///< bra prim index.
    int ub = 0;   ///< ket prim index.
    double k = 0; ///< Gaussian product prefactor (geometry-fixed).
};

/// @brief Device P→weight pack + acc→V scatter workspace.
///
/// One instance owns prim/contrib tables and an nb×nb V buffer. Product
/// geometry arrays and `DeviceProduct` buffers are owned by the caller
/// (per level under multigrid; α/β under UKS) and passed each launch.
class ProductWeightPackDevice {
  public:
    ProductWeightPackDevice() = default;
    ~ProductWeightPackDevice() { destroy(); }

    ProductWeightPackDevice(const ProductWeightPackDevice&) = delete;
    ProductWeightPackDevice& operator=(const ProductWeightPackDevice&) = delete;

    /// @brief Allocate prim/contrib tables + V (nb×nb column-major).
    ///
    /// @param stream    Non-default SCF stream (caller-owned).
    /// @param nb        AO dimension.
    /// @param nprim     Length of prim table.
    /// @param ncontrib  Length of contrib pool.
    void build(cudaStream_t stream, int nb, int nprim, int ncontrib);

    /// @brief Upload geometry-fixed prim + contrib tables (once per run).
    void upload_tables(const DevicePrimRef* prims_host, int nprim,
                       const DevicePrimContrib* contribs_host, int ncontrib);

    /// @brief From device P (nb×nb column-major), write `d_prods[0..nprod)`.
    ///
    /// Weight w = k · Σ_ij ca_i P_μν cb_j; |w| < eps_drop → 0. Geometry
    /// fields are copied from `d_geom` (device-resident). Sets `mag = |w|`
    /// (per-monom); call `apply_group_max_mag` for collint group windows.
    ///
    /// @param d_P      Device density, length nb*nb, column-major [e].
    /// @param d_geom   Device product geometry, length nprod.
    /// @param d_prods  Device product descriptors (weight field written).
    /// @param nprod    Number of products.
    /// @param eps_drop Drop threshold on |w| (same as host mass margin).
    void pack_weights(const double* d_P, const DeviceProductGeom* d_geom,
                      DeviceProduct* d_prods, int nprod, double eps_drop);

    /// @brief Set each monom's `mag` + `L_win` to collint group maxima.
    ///
    /// Twin of CPU collint group window sizing: max live \|w\| and max monom
    /// Ltot on a geometry batch (same pol table). Without group L, monom
    /// continuous spheres under-size s-character monoms in mixed p/d groups
    /// vs collint `group_angular_max` (cuda-gpw-multigrid-device-ks residual).
    /// Tasks from `ProductSet::tasks_by_level`; uploaded once per run.
    ///
    /// @param d_prods    Device products (weight already packed).
    /// @param d_task_beg Device task begin indices, length ntask.
    /// @param d_task_end Device task end indices (exclusive), length ntask.
    /// @param ntask      Number of collint geometry tasks.
    void apply_group_max_mag(DeviceProduct* d_prods, const int* d_task_beg,
                             const int* d_task_end, int ntask);

    /// @brief Zero internal V then scatter product integrals into V_μν.
    ///
    /// Twin of host scatter: V(μ,ν) += k · w_grid · acc · ca · cb.
    /// Sequential product order (host-identical; not atomicAdd).
    ///
    /// @param d_geom  Device product geometry (ua/ub/k), length nprod.
    /// @param d_acc   Bare ∫ V·shape per product, length nprod.
    /// @param nprod   Number of products.
    /// @param w_grid  Grid weight V_cell/npts [Bohr³].
    void scatter_acc(const DeviceProductGeom* d_geom, const double* d_acc,
                     int nprod, double w_grid);

    /// @brief Accumulate scatter into existing internal V (no zero).
    ///
    /// Used for multigrid: sum per-level V contributions on device.
    void scatter_acc_add(const DeviceProductGeom* d_geom, const double* d_acc,
                         int nprod, double w_grid);

    /// @brief Download internal V (nb×nb column-major) to host.
    void download_V(double* V_host) const;

    /// @brief Frobenius ⟨P, M⟩ = Σ_ij P_ij M_ij (column-major device buffers).
    ///
    /// Used for mid-SCF e_kinetic / e_loc_sr without downloading P.
    double frobenius(const double* d_P, const double* d_M) const;

    /// @brief Upload a host nb×nb matrix to a caller-owned device buffer.
    void upload_matrix(const double* M_host, double* d_M) const;

    /// @brief Free device memory.
    void destroy();

    bool built() const { return built_; }
    int nb() const { return nb_; }
    double* d_V() { return d_V_; }
    const double* d_V() const { return d_V_; }
    DevicePrimRef* d_prims() { return d_prims_; }
    DevicePrimContrib* d_contribs() { return d_contribs_; }

  private:
    void ensure_built_() const;
    void scatter_acc_(const DeviceProductGeom* d_geom, const double* d_acc,
                      int nprod, double w_grid, bool zero_first);

    cudaStream_t stream_ = nullptr;
    int nb_ = 0;
    int nprim_ = 0;
    int ncontrib_ = 0;
    bool built_ = false;

    DevicePrimRef* d_prims_ = nullptr;
    DevicePrimContrib* d_contribs_ = nullptr;
    double* d_V_ = nullptr;      ///< nb×nb column-major.
    double* d_reduce_ = nullptr; ///< single double for frobenius reduction.
};

/// @brief Host→device geometry conversion (one product).
inline DeviceProductGeom to_device_geom(double p, const double pc[3],
                                        const double ra[3], const double rb[3],
                                        const int la[3], const int lb[3],
                                        int ua, int ub, double k) {
    DeviceProductGeom g;
    g.p = p;
    g.ua = ua;
    g.ub = ub;
    g.k = k;
    for (int d = 0; d < 3; ++d) {
        g.pc[d] = pc[d];
        g.ra[d] = ra[d];
        g.rb[d] = rb[d];
        g.la[d] = la[d];
        g.lb[d] = lb[d];
    }
    return g;
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
