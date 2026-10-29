/// @file cuda/longrange/pme.cuh
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Requires `ATV_BUILD_KSPACE`.
///
/// GPU mirror of `cpu::KSpacePME` (Essmann smooth PME): B-spline charge
/// spreading → cuFFT → influence-function multiply → inverse cuFFT → force
/// interpolation. Drop-in reciprocal replacement for `cuda::KSpaceEwald`;
/// shares α with `cuda::CoulLong`. Single-GPU.
///
/// One build fixes the FFT dimensions (chosen from `grid_spacing` and the
/// build-time cell), the spline order, α and the atom set; changing any of
/// them requires a rebuild. The cell may change between steps (NPT, deform):
/// every kernel reads the published device `Box` (geometry, volume and the
/// normalization derived from it), and the influence/virial coefficient
/// tables, at fixed device addresses, remember the `Box::revision` they were
/// computed for; the captured graph recomputes them in stream order when the
/// revision moves. A box change therefore needs neither graph recapture nor
/// new FFT plans. The mesh is NOT re-chosen when the cell
/// changes, so the grid spacing (and the reciprocal error) drift with the
/// box.
///
/// When stress is required the six-component reciprocal virial is the exact
/// strain derivative of the discrete PME energy at fixed fractional
/// coordinates, averaged over both conjugate half-spectrum indices (exact at
/// triclinic Nyquist modes); the background energy contributes its volume
/// derivative on the diagonal.
///
/// Spline/interpolation, real mesh and half-spectrum follow CUDA force_t
/// precision; geometry, influence coefficients and energy/virial reductions
/// remain double.
#pragma once

#include <array>
#include <string>

#include "atomiverse/core/precision.h"
#include "atomiverse/cuda/longrange/kspace.cuh"

namespace ATV {
namespace cuda {

/// @brief Which influence and differentiation the shared mesh path uses.
enum class ReciprocalKind {
    /// Essmann smooth PME: Euler-B influence, analytic B-spline derivatives.
    Spme,
    /// Hockney–Eastwood PPPM, ik differentiation (LAMMPS `pppm` default).
    PppmIk
};

/// @brief Smooth Particle-Mesh Ewald reciprocal electrostatics (CUDA).
///
/// `ReciprocalKind::PppmIk` keeps this class's spread, forward cuFFT and graph.
/// The three ik fields are one batched inverse transform, then one θ gather.
/// `cuda::KSpacePPPM` is that setting.
class KSpacePME : public KSpace {
  public:
    /// @param alpha        Ewald splitting parameter α [1/Å] (match CoulLong).
    /// @param grid_spacing Target real-space grid spacing [Å] used to choose
    ///                     the FFT dimensions at build. Default 1.0.
    /// @param order        B-spline order in [3, 12]. Default 6.
    /// @param kind         SPME, or ik PPPM. Default SPME.
    explicit KSpacePME(double alpha, double grid_spacing = 1.0, int order = 6,
                       ReciprocalKind kind = ReciprocalKind::Spme);
    ~KSpacePME() override;

    std::string name() const override { return "pme"; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, AccReal(alpha_), AccReal(0),
                AccReal(0)};
    }

    /// @brief Gate the reciprocal virial (set before build).
    /// @param required True iff a downstream consumer reads stress.
    void set_stress_required(bool required) override { stress_ = required; }

    /// @brief FFT dimensions chosen at build.
    /// @return {K0, K1, K2} [grid points]; zero before build.
    std::array<int, 3> mesh() const { return {K_[0], K_[1], K_[2]}; }

    // Charge-dependent background data (PME also captures it in a graph)
    // is fixed at build; reject edits until that cache has an update path.
    bool supports_input_changes(InputChange changes) const override {
        return !has_input_change(changes, InputChange::Identity)
               && !has_input_change(changes, InputChange::Charge);
    }
    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void enqueue_(const LaunchContext& ctx);
    void* graph_exec_ = nullptr; ///< cudaGraphExec_t, allocated only in build.
    int built_atoms_ = 0;
    double alpha_;
    double spacing_;
    int order_;
    ReciprocalKind kind_ = ReciprocalKind::Spme;
    bool stress_ = true;
    double background_coef_ = 0; ///< k_e π Q² / (2α²) [eV·Å³].

    int K_[3] = {0, 0, 0};
    int ntot_ = 0;
    int nhalf_ = 0;

    void* d_tables_ = nullptr;      ///< Coefficient-table revision (PmeTables).
    double* d_moduli_ = nullptr;    ///< B-spline moduli, K0 + K1 + K2 entries.
    double* d_gf_b_ = nullptr;      ///< PPPM denominator coefficients, `order`.
    void* d_potential_ = nullptr;   ///< PPPM V̂ half-spectrum, nhalf.
    double* d_influence_ = nullptr; ///< Conjugate-averaged influence, nhalf.
    double* d_virial_ = nullptr;    ///< Six virial coefficient planes, 6·nhalf.
    void* d_grid_ = nullptr; ///< Real mesh. SPME: ntot. PPPM: 3·ntot, batched.
    void* d_spectrum_ =
        nullptr; ///< Half-spectrum. SPME: nhalf. PPPM: 3·nhalf, batched.
    int forward_plan_ = -1; ///< R2C/D2Z cufftHandle; -1 when uncreated.
    int inverse_plan_ = -1; ///< C2R/Z2D. PPPM is batch 3; -1 when uncreated.
};

/// @brief Hockney–Eastwood ik PPPM on the shared CUDA mesh path.
class KSpacePPPM : public KSpacePME {
  public:
    /// @param alpha        Ewald α [1/Å], the same value as CoulLong.
    /// @param grid_spacing Target mesh spacing [Å]. Factors 2, 3 and 5 only.
    /// @param order        B-spline order in [3, 12]. Default 6.
    explicit KSpacePPPM(double alpha, double grid_spacing = 1.0, int order = 6)
        : KSpacePME(alpha, grid_spacing, order, ReciprocalKind::PppmIk) {}

    std::string name() const override { return "pppm"; }
};

} // namespace cuda
} // namespace ATV
