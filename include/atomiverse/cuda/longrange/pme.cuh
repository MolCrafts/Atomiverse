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
/// changes (LAMMPS `PPPM::setup` keeps `nx_pppm` and rebuilds only the
/// Green's function), so the grid spacing and the reciprocal error drift
/// with the box.
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

#include "atomiverse/core/pppm_accuracy.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cuda/device/graph_ring.cuh"
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
    KSpacePME(KSpacePME&&) noexcept = default;
    KSpacePME& operator=(KSpacePME&&) noexcept = default;

    std::string name() const override { return "pme"; }
    bool graph_capturable() const override { return true; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, AccReal(alpha_), AccReal(0),
                AccReal(0)};
    }

    /// @brief Gate the reciprocal virial (set before build).
    /// @param required True iff a downstream consumer reads stress.

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

  protected:
    /// Explicit mesh (`kspace_modify mesh`); zeros = derive from spacing.
    /// Set by `cuda::KSpacePPPM`'s mesh constructor only.
    int fixed_K_[3] = {0, 0, 0};

  private:
    void enqueue_(const LaunchContext& ctx, bool want_energy, bool want_virial);
    /// Graph per (eflag | vflag << 1), captured and instantiated in build.
    GraphRing graph_exec_[4];
    void* graphs_[4] = {}; ///< cudaGraph_t of each variant (for splicing).
    int built_atoms_ = 0;
    double alpha_;
    double spacing_;
    int order_;
    ReciprocalKind kind_ = ReciprocalKind::Spme;
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
    double* d_ik_ =
        nullptr; ///< PPPM folded ik planes (pppm_half_mode), 3·nhalf.
    void* d_grid_ = nullptr; ///< Real mesh. SPME: ntot. PPPM: 3·ntot, batched.
    void* d_spectrum_ =
        nullptr; ///< Half-spectrum. SPME: nhalf. PPPM: 3·nhalf, batched.
    int forward_plan_ = -1; ///< R2C/D2Z cufftHandle; -1 when uncreated.
    int inverse_plan_ = -1; ///< C2R/Z2D. PPPM is batch 3; -1 when uncreated.
};

/// @brief Hockney–Eastwood ik PPPM on the shared CUDA mesh path.
///
/// Same parameters, defaults and mesh rule as `cpu::KSpacePPPM`: the mesh is
/// ⌈|a_d| / grid_spacing⌉ (or the explicit mesh) rounded up to sizes with
/// factors 2, 3 and 5 (LAMMPS `factorable`), at least order + 1; charges are
/// spread and the field is gathered on LAMMPS's centred stencil
/// (`core/pppm_ik.h`), not on the one-sided smooth-PME stencil of
/// `KSpacePME`. The half-spectrum energy, virial and ik field fold each mode
/// with its conjugate partner (`pppm_half_mode`), LAMMPS's full complex
/// `poisson_ik` on every mesh, even ones included.
///
/// Every atom is wrapped into the cell before it is mapped and the mesh is
/// replicated on the one device, so there is no ghost depth an atom can
/// outrun between reneighbourings (the CPU brick's `0.5·skin` bound); a
/// non-finite coordinate is not deposited and its non-finite force raises
/// the frame's force fault in the gather.
class KSpacePPPM : public KSpacePME {
  public:
    /// @param alpha        Ewald α [1/Å], the same value as CoulLong.
    /// @param grid_spacing Target mesh spacing [Å]. Factors 2, 3 and 5 only.
    /// @param order        Charge-assignment order in [3, 12]. Default 6.
    explicit KSpacePPPM(double alpha, double grid_spacing = 1.0, int order = 6)
        : KSpacePME(alpha, grid_spacing, order, ReciprocalKind::PppmIk) {}

    /// @brief Construct on an explicit mesh (LAMMPS `kspace_modify mesh`),
    ///        the same rule as `cpu::KSpacePPPM`'s mesh constructor.
    /// @throws std::invalid_argument on a non-positive mesh entry.
    KSpacePPPM(double alpha, std::array<int, 3> mesh, int order = 6);

    /// @brief α and mesh from an accuracy target (LAMMPS `kspace_style pppm
    ///        <accuracy>`), `longrange::pppm_from_accuracy` — the same
    ///        request gives the same solver as `cpu::KSpacePPPM`.
    static KSpacePPPM from_accuracy(const longrange::PppmAccuracyRequest& req) {
        const auto r = longrange::pppm_from_accuracy(req);
        return KSpacePPPM(r.alpha, {r.mesh[0], r.mesh[1], r.mesh[2]}, r.order);
    }

    std::string name() const override { return "pppm"; }
};

} // namespace cuda
} // namespace ATV
