// Private LSDA/PBE-family RKS and spin frozen-density operators; exact existing PW quadrature/law.
#pragma once
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/fft.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_operator_virial.cuh"
namespace ATV::cuda::pw::internal {

struct FrozenLsdaWorkspaceView {
    cufftDoubleComplex* scaled_density; // ng, leaves original rho_orb immutable
    cufftDoubleComplex* half;           // XC FFT nc
    double *rho, *v, *epsilon;          // XC FFT nr each
    double *energy, *energy_partials, *virial_partials; // 1,256,256
    FrozenOperatorVirial* output;                       // 1, overwritten
};
// All pointers are device views, plans and stream are cold-owned by the caller.
// omega is the deformed volume, density_scale=Omega_reference/omega. No arrays
// cross the host boundary in this evaluation. Uses the real production LSDA
// grid law and xc_energy, not a rewritten VWN model or the PBE sigma=0 limit.
void evaluate_lsda_frozen_virial(FFTPlan3D& quadrature, const int* h,
                                 const int* k, const int* l, int ng,
                                 const cufftDoubleComplex* rho_orb,
                                 double density_scale, double omega,
                                 FrozenLsdaWorkspaceView, cudaStream_t);

struct FrozenPbeWorkspaceView {
    cufftDoubleComplex *scaled_density, *derivative, *half; // ng,ng,nc
    double *rho, *v, *epsilon; // nr each; v is the LOCAL point potential
    double *raw_gradient[3], *flux[3]; // nr each, mutually disjoint
    double *energy, *energy_partials, *virial_partials; //1,256,9*256
    FrozenOperatorVirial* output;
};
// Same fixed-hkl density scaling as LSDA, with physical G = F^-T G_ref.
// Uses the actual >=3x PW GGA quadrature and existing xc_pbe_grid/xc_energy.
// The caller owns matching FFT-plan/stream binding (FFTPlan3D has no stream
// getter), disjoint scratch and immutable input lifetimes. No allocation,
// host transfer or synchronization; three nr D2D copies retain raw gradients
// before the production point kernel overwrites flux. FFT c2r also preserves
// its nc input with a D2D copy. W9 is unprojected and includes isotropic grad·flux.
void evaluate_pbe_frozen_virial(cpu::xc::XCType, FFTPlan3D&, const int* h,
                                const int* k, const int* l, const double* gx,
                                const double* gy, const double* gz, int ng,
                                const cufftDoubleComplex* density,
                                double density_scale, double omega,
                                FrozenPbeWorkspaceView, cudaStream_t);

struct FrozenSpinWorkspaceView {
    cufftDoubleComplex *scaled_density[2], *derivative, *half;
    double *rho[2], *v[2], *epsilon;
    double *raw_gradient[2][3], *flux[2][3];
    double *energy, *energy_partials, *virial_partials;
    FrozenOperatorVirial* output;
};
// Positive-spin LSDA/PBE-family pure tensor; negative/floor endpoints are not
// qualified by the current evidence. Both spin slots and all scaled/rho/v/
// raw-gradient/flux buffers are mutually disjoint and do not overlap the
// immutable inputs. Caller owns scratch and FFT bound to this exact stream.
// No allocations, host copies or synchronization. GGA: six nr D2D gradient
// copies plus the existing FFT input-preservation copies are explicit.
void evaluate_spin_frozen_virial(cpu::xc::XCType, FFTPlan3D&, const int*,
                                 const int*, const int*, const double*,
                                 const double*, const double*, int,
                                 const cufftDoubleComplex*,
                                 const cufftDoubleComplex*, double, double,
                                 FrozenSpinWorkspaceView, cudaStream_t);

} // namespace ATV::cuda::pw::internal
