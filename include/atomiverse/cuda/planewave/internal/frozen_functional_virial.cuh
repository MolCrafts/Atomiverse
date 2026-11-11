#pragma once
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_operator_virial.cuh"
#include <memory>
#include <optional>
namespace ATV::cuda::pw::internal {
struct FrozenDerivativeCertificate {
    double max_antisymmetric_ha = 0, density_mismatch_squared = 0;
    int finite = 1, reference_unchanged = 1;
};
// Pure device reduction: no allocations, copies or synchronization.
void certify_frozen_derivatives(
    const FrozenOperatorVirial* terms, const double* forces,
    const cufftDoubleComplex* reference, const cufftDoubleComplex* current,
    const cufftDoubleComplex* orbital, int ng0, int atoms, bool force,
    bool stress, FrozenDerivativeCertificate* out, double force_scale,
    double hartree_ev, double volume, cudaStream_t);
enum class FrozenFinishResult { complete, unsupported_domain };
struct FrozenSpinDomainInput {
    const double* rho[2]{};
    const double* gradient[2][3]{};
    int count = 0;
    bool gga = false;
};
struct FrozenSpinDomainStatus {
    double minimum[2]{}, max_abs_zeta = 0;
    unsigned violations = 0, finite = 1;
};
// No allocation, transfers or synchronization. Inputs are the raw orbital XC
// quadrature already owned for stress; caller owns one scalar status packet.
void classify_frozen_spin_domain(FrozenSpinDomainInput, FrozenSpinDomainStatus*,
                                 cudaStream_t);
struct FixedSpinOccupations {
    int alpha = 0, beta = 0;
};
struct FrozenFunctionalOutcome {
    FinalBandStatus auxiliary;
    double auxiliary_mu_ha = 0, auxiliary_entropy = 0,
           auxiliary_electron_count = 0;
    FrozenDerivativeCertificate certificate;
    std::size_t explicit_device_bytes = 0;
    bool forces_requested = false, stress_requested = false, computed = false,
         published = false;
    FrozenSpinDomainStatus spin_domain{};
    bool spin_domain_evaluated = false, legacy_fallback = false;
    int spin_channels = 1;
    std::array<FinalBandStatus, 2> spin_auxiliary{};
    std::array<double, 2> fixed_device_f_count{}, spin_orbital_electrons{},
        spin_rho_G0_electrons{};
    bool fixed_spin_prefix_exact = false;
};
// Method-private bounded read-only views. Buffers are owned by this workspace;
// all views expire before solver/density/borrowed stream destruction. Never Frame state.
struct FrozenCandidateView {
    FrozenBandDeviceView orbitals;
    const cufftDoubleComplex *reference_density, *orbital_density;
    const FrozenOperatorVirial* terms; //7 only when stress requested
    const double* forces;              //3N only when forces requested
    FrozenFunctionalOutcome outcome;
    std::array<const cufftDoubleComplex*, 2> reference_spin_density{},
        orbital_spin_density{};
    std::optional<FixedSpinOccupations> fixed_spin;
    int live_bands = 0;
};
// Gamma RKS fixed/FD and fixed nonempty UKS LSDA/PBE-family only. The caller skips construction entirely for energy-only.
// Cold radial/Ewald preparation is host work. finish performs one fresh final
// band sweep plus its actual Hpsi certification, then requested force/virial work.
// No profile knobs, diagnostic callbacks or SCF loop. The supported analytic
// branch performs no orbital/density host readback; requested host-output
// boundaries remain explicit. A finite unsupported XC domain delegates to the
// inherited numerical epilogues with their host readbacks and staged F/W H2D.
class FrozenFunctionalWorkspace {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    FrozenFunctionalWorkspace(
        const cpu::Frame&, const cpu::pw::internal::PWIons&,
        const cpu::pw::internal::GVectorList& density_modes,
        const cpu::pw::internal::GVectorList& gamma_modes, int bands,
        cpu::xc::XCType, electronic::Smearing, double width_ha,
        std::optional<FixedSpinOccupations>, cudaStream_t, bool forces,
        bool stress);
    ~FrozenFunctionalWorkspace();
    FrozenFinishResult finish(DeviceDensity&, DeviceBandSolver&,
                              xt::xarray<AccReal>& eigenvalues,
                              const cpu::pw::PWResult&,
                              std::vector<std::array<double, 3>>* host_force,
                              std::array<double, 6>* host_stress,
                              cuda::Frame* parent);
    // Finite unsupported domains use the inherited numerical epilogues with
    // the same accepted solve. Stage all completed legacy candidates first;
    // this method validates and commits them together, never another SCF.
    void commit_fallback(DeviceDensity&, const cpu::pw::PWResult&,
                         const std::vector<std::array<double, 3>>& force_ev,
                         const std::array<double, 6>& sigma_ev_ang3,
                         cuda::Frame* parent);
    FrozenCandidateView candidate() const;
};
} // namespace ATV::cuda::pw::internal
