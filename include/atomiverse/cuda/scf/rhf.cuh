/// @file cuda/scf/rhf.cuh
/// @brief cuda::scf::RHF — concrete restricted Hartree–Fock (CUDA backend).
///
/// Mirror of `cpu::scf::RHF`: **constructor arguments** for parameters,
/// **Frame** as the only public I/O. Construct with
/// `cuda::scf::RHF{cpu::scf::BasisSet::STO3G}` (optional charge / tol /
/// max_iter). No freestanding integral bag; no `rhf.basis = …`.
///
/// **Energy (CUDA-only HF for STO-3G s/p):** device integral kernels + SCF
/// (`launch_*_sp`) — no host ERI. **Forces:** s/p device analytic
/// (`launch_rhf_forces_sp`); higher L still host.
///
/// Multi-step BOMD: `cuda::md::MD` + `set_evaluator(RHF{…})` + VelocityVerlet.
///
/// Units: `kEnergy` [eV]; forces `kAtom_f*` [eV/Å]; `energy()` [Hartree].
///
/// @code{.cpp}
///   cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
///   rhf.with_gradient(false);       // energy-only: stay on device for H2
///   rhf.run(frame);                 // meta kEnergy [eV]
///
///   cuda::md::MD sim;
///   sim.set_integrator(cuda::md::VelocityVerlet{0.5});
///   sim.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
///   sim.build();
///   sim.run(frame, 3);
/// @endcode
#pragma once

#include <memory>
#include <string>
#include <utility>

#include "atomiverse/core/role.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {

// Incomplete type — defined only in the .cu translation unit (private).
class SCFSolver;

/// @brief Restricted Hartree–Fock on the CUDA backend.
///
/// Closed-shell RHF. Construct with basis (and optional SCF knobs); Frame
/// in/out only. Private device workspace is owned by this object.
class RHF : public cuda::Component, public cuda::HF {
  public:
    /// @brief Basis set (set at construction only).
    const cpu::scf::BasisSet basis;
    /// @brief Molecular charge [e] (electrons = ΣZ − charge).
    const int charge;
    /// @brief Density residual Frobenius tolerance [dimensionless].
    const double tol;
    /// @brief Maximum SCF iterations [dimensionless].
    const int max_iter;
    /// @brief Virtual-orbital level shift λ [Hartree] (0 = off).
    const double level_shift;

    /// @brief When true, nested Evaluate / launch may assemble nuclear forces
    ///        after SCF (standalone `forces()` always computes them).
    ///        Mutable for `with_gradient` chaining only.
    bool compute_force = true;

    /// @brief Construct RHF parameters (primary public form).
    /// @param basis_in Basis set (e.g. `BasisSet::STO3G`).
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Residual tolerance [dimensionless].
    /// @param max_iter_in Max SCF iterations [dimensionless].
    /// @param level_shift_in Virtual level shift [Hartree].
    /// @param compute_force_in Nested force assembly default.
    explicit RHF(cpu::scf::BasisSet basis_in = cpu::scf::BasisSet::STO3G,
                 int charge_in = 0, double tol_in = 1e-9,
                 int max_iter_in = 100, double level_shift_in = 0.0,
                 bool compute_force_in = true);

    /// @brief Destroy owned internal solver (defined in .cu).
    ~RHF() override;
    /// @brief Move (const parameter fields are moved via construction).
    RHF(RHF&&) noexcept;
    RHF& operator=(RHF&&) = delete;
    RHF(const RHF&) = delete;
    RHF& operator=(const RHF&) = delete;

    /// @brief Enable / disable force assembly for nested Evaluate paths.
    RHF& with_gradient(bool enabled) & noexcept {
        compute_force = enabled;
        return *this;
    }
    /// @brief Enable / disable force assembly (rvalue chain).
    RHF&& with_gradient(bool enabled) && noexcept {
        compute_force = enabled;
        return std::move(*this);
    }

    /// @brief Component name.
    /// @return `"rhf"`.
    std::string name() const override { return "rhf"; }

    Role role() const override { return 0; }

    /// @brief One-shot SCF on a host Frame (H2D → solve → D2H via Evaluate).
    /// @param frame System state [Å, Z]; energy written back [eV on meta].
    void run(cpu::Frame& frame);

    /// @brief Analytic nuclear forces after a device SCF.
    /// @param frame Geometry [Å, Z]; forces [eV/Å] and energy [eV] written.
    /// @return Forces [eV/Å] and energy [eV].
    cpu::scf::GradientResult forces(cpu::Frame& frame);

    /// @brief Last total energy [Hartree].
    double energy() const { return energy_ha_; }

    /// @brief Whether the most recent SCF reached `tol`.
    bool converged() const { return converged_; }

    /// @brief SCF iterations of the most recent solve.
    int iterations() const { return iterations_; }

    /// @brief True when the most recent force assembly used device kernels
    ///        (s/p STO-3G path). False for host fallback or energy-only.
    bool last_force_on_device() const { return last_force_device_; }

    /// @brief Device SCF workspace after a successful `launch()` / `run()`.
    const SCFWorkspace& workspace() const;

    /// @brief Mark ready for multi-step MD / Evaluate (no geometry freeze).
    void build(const BuildContext& ctx) override;

    /// @brief SCF (+ optional forces) from current Frame geometry.
    void launch(const LaunchContext& ctx) override;

    /// @brief Release private device resources.
    void destroy(const DestroyContext& ctx) override;

  private:
    std::unique_ptr<SCFSolver> solver_;
    double energy_ha_ = 0.0;
    bool converged_ = false;
    int iterations_ = 0;
    bool built_ = false;
    bool last_force_device_ = false;
};

} // namespace scf
} // namespace cuda
} // namespace ATV
