/// Fixed-Miller PW device operator energies and cell virials. No SCF or total stress.
#pragma once
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV::cuda::pw::internal {
// Borrow device storage and the method/backend's current stream. Geometry-only
// radial jets may be constructed from immutable CPU NcPotentialData and uploaded
// during cold setup; that setup is not a device-only path. No Driver/SCF state
// is owned here. All scalars and tables use FP64 physical atomic units.
//
// Deformation keeps C, f and Miller modes fixed: q'=F^-T q, R'=F R,
// Omega'=det(F)Omega, rho'_G=rho_G/det(F). Density G lists are Gamma-centred;
// wave-vector lists include their k shift. No Gamma doubling is applied.
//
// Full W_ab=-dE/depsilon_ab in Hartree. Keep all nine diagnostic lanes;
// never symmetrize away a missing angular derivative or phase mistake.
struct FrozenOperatorVirial {
    double energy_ha;
    double virial_ha[9];
};
struct FrozenReciprocalView {
    const double *x, *y, *z;
    int count;
};
struct FrozenBandDeviceView {
    const cufftDoubleComplex* C; // column-major countG x bands, normalized
    const double* f;             // signed, includes spin factor exactly once
    int bands;
    double k_weight; // use as supplied; never abs/positive-only screening
};
struct FrozenLocalDeviceView {
    const double* value; // v(g)=U(g)/Omega, including production G0 finite part
    const double* slope; // dv/dg at fixed Omega, exact same radial quadrature
    const cufftDoubleComplex* structure; // sum_atoms exp(-i G.R), cold geometry
};
struct FrozenProjectorColumn {
    int atom, l, m, projector;
    int block_first,
        block_size; // same atom,l,m; all radial projectors in block
};
struct FrozenNonlocalDeviceGeometry {
    FrozenReciprocalView q;     // k+G, not the density sphere G
    const double* R;            // raw radial regularized R_l(q), [projector][G]
    const double* slope_over_q; // R'_l(q)/q, regular also at q=0
    const double* D; // immutable original full symmetric D_ij, Ha, row-major
    const double* positions; // [atom][xyz], Bohr
    const FrozenProjectorColumn* columns;
    int projectors, column_count;
    double prefactor; // 4*pi/sqrt(Omega), included exactly once
};
struct FrozenNonlocalDeviceScratch {
    cufftDoubleComplex*
        B; // countG x column_count; also reused for eta operator
    cufftDoubleComplex* c; // column_count x bands
    cufftDoubleComplex* Dc;
    cufftDoubleComplex* eta;
};
// Caller zeros each output before a complete accumulation. The complete original
// symmetric D is used, including signed/off-diagonal elements and every radial
// prefix. There is no amplitude or occupation screening. As for the corresponding
// CPU operator, arbitrary exponent rescaling requires representable intermediates.
// Hartree alone retains the existing Poisson g^2<1e-16 zero-mode exclusion.
//
// Geometry columns must be cold-validated: l in [0,2], m in [0,2l], valid atom and
// projector indices, and each contiguous block contains exactly the radial
// projectors of one (atom,l,m). D has projectors^2 entries; R and slope tables
// each have projectors*countG; positions have 3*atom_count. No per-atom dense
// derivative tensor is retained.
// Scratch extents: B=countG*column_count, each of c/Dc/eta=column_count*bands.
// Scratch and output must not alias each other or the read-only inputs. The
// supplied cuBLAS handle must already use the supplied stream and HOST pointer
// mode; it is never reconfigured here. Calls validate host shapes/metadata before
// launching, then enqueue asynchronously; the caller owns the final error fence.
// Borrowed inputs,
// outputs, BLAS handle and stream must remain live until the stream finishes.
// No allocations or host/device copies occur in these operator calls.
void kinetic_frozen_virial(FrozenReciprocalView q, FrozenBandDeviceView state,
                           FrozenOperatorVirial* output, cudaStream_t);
void hartree_frozen_virial(FrozenReciprocalView G,
                           const cufftDoubleComplex* rho, double omega,
                           FrozenOperatorVirial* output, cudaStream_t);
void local_potential_frozen_virial(FrozenReciprocalView G,
                                   const cufftDoubleComplex* rho, double omega,
                                   FrozenLocalDeviceView geometry,
                                   FrozenOperatorVirial* output, cudaStream_t);
// Shared projection for force and tensor consumers. Packed rows may contain
// independent spin blocks: no cross-band or cross-spin contractions occur, and
// no inter-spin orthogonality or bands<=countG restriction is imposed here.
// project initializes value B, c and Dc. The force consumer adds to caller-zeroed
// forces and requires that value B; it leaves the shared projection reusable.
// add_projected_nonlocal_virial overwrites B with derivative columns. Consume
// force before this tensor call, or explicitly re-project before a later force.
// qC scratch extent is countG*bands; the same nonalias/stream/lifetime contracts
// above apply. Calls issue no explicit allocation, memcpy or synchronization;
// vendor-library internal behavior is outside this source-level contract.
// The force consumer requires nonempty q/bands and a successful projection
// (column_count==0 is a no-op); zero-q/zero-band force calls are unqualified.
void project_frozen_nonlocal(FrozenNonlocalDeviceGeometry geometry,
                             FrozenBandDeviceView state,
                             FrozenNonlocalDeviceScratch scratch,
                             cublasHandle_t blas, cudaStream_t stream);
void accumulate_projected_nonlocal_forces(FrozenNonlocalDeviceGeometry geometry,
                                          FrozenBandDeviceView state,
                                          FrozenNonlocalDeviceScratch scratch,
                                          cufftDoubleComplex* qC,
                                          cublasHandle_t blas, double* force,
                                          cudaStream_t stream);
void add_projected_nonlocal_virial(FrozenNonlocalDeviceGeometry geometry,
                                   FrozenBandDeviceView state,
                                   FrozenNonlocalDeviceScratch scratch,
                                   cublasHandle_t blas,
                                   FrozenOperatorVirial* output,
                                   cudaStream_t stream);
void nonlocal_frozen_virial(FrozenNonlocalDeviceGeometry geometry,
                            FrozenBandDeviceView state,
                            FrozenNonlocalDeviceScratch scratch,
                            cublasHandle_t blas, FrozenOperatorVirial* output,
                            cudaStream_t stream);
} // namespace ATV::cuda::pw::internal
