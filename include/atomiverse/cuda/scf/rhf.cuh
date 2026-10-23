/// @file cuda/scf/rhf.cuh
/// @brief cuda::scf::RHF — concrete restricted Hartree–Fock (CUDA backend).
///
/// Mirror of `cpu::scf::RHF`: **constructor arguments** for parameters,
/// **Frame** as the only public I/O. The SCF iteration is
/// `cuda::Driver<scf::Phase>` (`fock → mix → diagonalize → density →
/// check`), the same stage order as `MeanFieldMethod`. PySCF
/// `scf.hf.kernel` is that loop (`get_fock` / `eig` / `make_rdm1` /
/// `energy_tot` / convergence). `energy_elec` is the electronic piece
/// (½ Tr((H+F)P)); `energy_tot` adds the nuclear repulsion.
/// `scf.diis.CDIIS.update` is the mix stage.
///
/// Not a Component. Outer `cuda::Evaluate` / `cuda::md::MD` nest it with
/// `set_evaluator` / `on(phase).add`, which mounts a private slot.
///
/// **Energy:** device integrals, no host ERI. STO-3G l ≤ 1 uses
/// `launch_*_sp`. Cartesian l ≤ 3 uses `launch_*_lgt1`. Spherical bases
/// with l ≤ 3 (def2-SVP, def2-TZVP) use that Cartesian engine and the
/// CPU CartToSph sandwich on device. l ≥ 4 throws `launch_eri_g`.
/// **Forces:** STO-3G s/p (`launch_rhf_forces_sp`). Any other basis
/// throws `launch_rhf_forces_lgt1` on `run(Frame&)` / `forces(Frame&)`
/// when `compute_force` is set — `with_gradient(false)` keeps that
/// energy. `run(const Frame&)` never assembles forces.
///
/// Units: `kEnergy` [eV] from `run(Frame&)` only; forces `kAtom_f*`
/// [eV/Å]; `energy()` [Hartree]. `run(const Frame&)` writes nothing.
///
/// @code{.cpp}
///   cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
///   rhf.run(frame);                 // overwrites kEnergy [eV]
///   const cpu::Frame& geometry = frame;
///   rhf.run(geometry);              // writes nothing; energy() [Ha]
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
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {

struct RhfLoopState;

/// @brief Restricted Hartree–Fock on the CUDA backend.
///
/// Closed-shell RHF. Construct with basis (and optional SCF knobs); Frame
/// in/out only. The self-consistency loop is this Driver. Private device
/// workspace is owned by this object and released when `run` / `forces`
/// returns.
class RHF : public cuda::Driver<ATV::scf::Phase>,
            public cuda::DriverNode,
            public cuda::HF {
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

    /// @brief When true, nested Evaluate / the SCF epilogue may assemble
    ///        nuclear forces after SCF (standalone `forces()` always
    ///        computes them). Mutable for `with_gradient` chaining only.
    bool compute_force = true;

    /// @brief Construct RHF parameters (primary public form).
    /// @param basis_in Basis set (e.g. `BasisSet::STO3G`).
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Residual tolerance [dimensionless].
    /// @param max_iter_in Max SCF iterations [dimensionless].
    /// @param level_shift_in Virtual level shift [Hartree].
    /// @param compute_force_in Nested force assembly default.
    explicit RHF(cpu::scf::BasisSet basis_in = cpu::scf::BasisSet::STO3G,
                 int charge_in = 0, double tol_in = 1e-9, int max_iter_in = 100,
                 double level_shift_in = 0.0, bool compute_force_in = true);

    /// @brief Destroy owned device SCF workspace (defined in .cu).
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

    /// @brief True when this method contributes a system energy.
    bool provides_energy() const override { return true; }
    /// @brief Force probe follows `compute_force` (`with_gradient`).
    bool provides_force() const override { return compute_force; }

    /// @brief Nested budget is the SCF iteration cap.
    /// @return `max_iter` [dimensionless].
    int default_budget() const override { return max_iter; }

    /// @brief Compile the stage Plan. Geometry-dependent integrals wait
    ///        until `run_nested_erased` (nuclei move between MD steps).
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;

    /// @brief One full SCF (+ optional forces) on the parent's device frame.
    /// @param ctx Parent CUDA context (stream + loop slot, restored on exit).
    /// @param frame Parent device frame. Energy [eV] and forces [eV/Å].
    /// @param budget SCF iterations; `< 0` or `0` uses `max_iter`.
    /// @return Iterations executed.
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;

    /// @brief Drop a live device workspace, if `run_nested` was interrupted.
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

    /// @brief One-shot SCF on a host Frame (H2D → staged SCF → D2H).
    ///
    /// PySCF `scf.hf.kernel`. The total is `energy_elec` + nuclear
    /// repulsion (`energy_tot`). Overwrites `kEnergy` [eV]. Hartree
    /// stays on `energy()`.
    /// @param frame System state [Å, Z]; energy written back [eV on meta].
    void run(cpu::Frame& frame);

    /// @brief Same SCF as `run(Frame&)`, geometry read-only.
    ///
    /// `kernel` / `energy_elec` / `energy_tot` still run. Nuclear forces
    /// are not assembled, and the device frame is not retrieved, so
    /// positions, forces, and meta (including `kEnergy`) are unchanged.
    /// @param frame System state [Å, Z]. Not written.
    void run(const cpu::Frame& frame);

    /// @brief Analytic nuclear forces after a device SCF.
    /// @param frame Geometry [Å, Z]; forces [eV/Å] and energy [eV] written.
    /// @return Forces [eV/Å] and energy [eV].
    cpu::scf::GradientResult forces(cpu::Frame& frame);

    /// @brief Last total energy [Hartree].
    double energy() const { return energy_ha_; }

    /// @brief SCF iterations of the most recent solve.
    int iterations() const { return iterations_; }

    /// @brief True when the most recent force assembly used device kernels
    ///        (s/p STO-3G path). False for energy-only. Out-of-coverage
    ///        `forces()` throws instead of a host fallback.
    bool last_force_on_device() const { return last_force_device_; }

    /// @brief True when the most recent SCF built integrals on device
    ///        (`launch_*_sp` or `launch_*_lgt1`), including a non-zero
    ///        level shift. False only if `run` / `forces` has not completed
    ///        a device SCF (a missing kernel throws instead of host-filling).
    bool last_scf_on_device() const { return last_device_sp_; }

    /// @brief Device SCF workspace during a solve.
    /// @throws std::runtime_error after `run` / `forces` return — the
    ///         workspace is released at the end of the call.
    const SCFWorkspace& workspace() const;

  protected:
    /// @brief The method is the energy source; no extra component is required.
    void validate_components() const override {}

  private:
    /// @brief Seed nuclear Z from a host Frame (path select before upload).
    void cache_host_types(const cpu::Frame& frame);

    /// @brief Bind fock/mix/diagonalize/density/check and the criterion.
    void wire_stages_();

    /// @brief SCF on an already-resident device frame.
    /// @param nuclear_forces Assemble device nuclear forces after the
    ///        loop. `run(Frame&)` and the nested mount pass
    ///        `compute_force`. `run(const Frame&)` passes false.
    /// @return Iterations executed.
    int drive_loop_(cuda::Context& ctx, cuda::Frame& frame, int budget,
                    bool nuclear_forces);

    /// @brief Walk this Driver's Plan once (one SCF iteration).
    void step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step);

    /// @brief Build or rebuild the private integral workspace at `frame`.
    /// @param nuclear_forces Refuse a basis with no device force kernel
    ///        before the SCF starts. Ignored by `run(const Frame&)`.
    void assemble_(cuda::Context& ctx, cuda::Frame& frame, bool nuclear_forces);

    /// @brief Sync, destroy components, shut the backend down. Does not
    ///        retrieve: `to_cpu` would overwrite host positions, forces,
    ///        and `kEnergy`.
    void release_backend_();

    /// @brief Free the private integral workspace, if any.
    void release_engine_();

    /// @brief Device s/p analytic forces into `frame` [eV/Å].
    void assemble_forces_(cuda::Context& ctx, cuda::Frame& frame);

    /// @brief One D2H of the device energy scalar into `energy_ha_` [Ha].
    void pull_energy_(cuda::Context& ctx, cuda::Frame& frame);

    std::unique_ptr<RhfLoopState> loop_;
    double energy_ha_ = 0.0;
    int iterations_ = 0;
    bool last_force_device_ = false;
    bool last_device_sp_ = false;
    /// Host-side nuclear Z [e] for path select + shell pack (no type[] D2H).
    std::vector<int> host_Z_;
};

} // namespace scf
} // namespace cuda
} // namespace ATV
