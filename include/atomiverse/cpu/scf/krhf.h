/// @file cpu/scf/krhf.h
/// @brief k-point restricted Hartree–Fock on an orthorhombic cell.
///
/// The SCF loop is `cpu::Driver<scf::Phase>` (`fock → mix → diagonalize →
/// density → check`), the same preset shape as `cpu::md::MD`. PySCF 2.14.0
/// `pyscf/pbc/scf/khf.py` (`KRHF`, `KSCF.get_fock`, `energy_elec`, `eig`,
/// `make_rdm1`) and `pyscf/scf/hf.py` `kernel` are that loop with DIIS left
/// off (`get_fock` `diis=None`). It is not a `MeanFieldMethod` (that class
/// stays molecular / Γ).
///
/// `run(Frame&)` overwrites `kEnergy` [eV]. `energy()` returns Hartree.
/// Analytic nuclear forces are not this product — `forces` throws
/// naming `krhf_nuclear_forces`.
#pragma once

#include <array>
#include <memory>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted Hartree–Fock with Monkhorst–Pack k-point sampling.
///
/// All-electron FFT Coulomb and exchange (PySCF `FFTDF`, exxdiv
/// `'ewald'`) for s-type Gaussians on an orthorhombic `Frame.box`.
/// A shell with l > 0 throws (`krhf_ao_lgt0`).
class KRHF : public cpu::Driver<ATV::scf::Phase>, public HF {
  public:
    /// @brief Basis set. Fixed at construction.
    const BasisSet basis;
    /// @brief Monkhorst–Pack divisions along a, b, c. Each entry ≥ 1.
    ///        `{1,1,1}` is the Γ point alone.
    const std::array<int, 3> k_mesh;
    /// @brief Cell charge [e]. Electrons per cell = ΣZ − charge;
    ///        the k-mesh total follows PySCF `tot_electrons(nk)`.
    const int charge;
    /// @brief |ΔE| convergence threshold [Hartree].
    const double tol;
    /// @brief Maximum SCF iterations.
    const int max_iter;

    /// @brief Construct k-point RHF.
    /// @param basis_in Gaussian basis (s functions only on this path).
    /// @param k_mesh_in Monkhorst–Pack mesh, each component ≥ 1.
    /// @param charge_in Cell charge [e].
    /// @param tol_in Energy change threshold [Hartree].
    /// @param max_iter_in Iteration cap.
    KRHF(BasisSet basis_in, std::array<int, 3> k_mesh_in, int charge_in = 0,
         double tol_in = 1e-9, int max_iter_in = 40);

    /// @brief Release a live SCF workspace. Defined in the .cpp where the
    ///        stage loop is complete.
    ~KRHF() override;

    /// @brief Move. Stage pointers address a heap loop and stay valid.
    KRHF(KRHF&&) noexcept;
    KRHF& operator=(KRHF&&) = delete;
    KRHF(const KRHF&) = delete;
    KRHF& operator=(const KRHF&) = delete;

    /// @brief This product contributes a total energy.
    bool provides_energy() const override { return true; }

    /// @brief This product has no analytic nuclear gradient.
    bool provides_force() const override { return false; }

    /// @brief Whole-cell evaluator: every atom on the rank.
    bool requires_full_frame() const override { return true; }

    /// @brief Nested budget is the SCF iteration cap.
    /// @return `max_iter` [dimensionless].
    int default_budget() const override { return max_iter; }

    /// @brief One SCF on the parent's Runtime. Accumulates `kEnergy` [eV].
    /// @param ctx Parent context. Loop status is restored on exit.
    /// @param frame Parent frame.
    /// @param budget SCF iterations; `<= 0` uses `max_iter`.
    /// @return Iterations executed.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;

    /// @brief Drop a cycle left open when a nested solve is interrupted.
    void destroy_on(cpu::Runtime& rt) override;

    /// @brief Run SCF. Overwrites `kEnergy` and `kEnergyElectronic` [eV].
    /// @param frame Orthorhombic cell, positions [Å], species in `kAtom_type`.
    void run(Frame& frame);

    /// @brief Run SCF without writing the Frame.
    /// @param frame Orthorhombic cell (read only).
    void run(const Frame& frame);

    /// @brief Analytic forces are not implemented.
    /// @param frame Unused.
    /// @return Never returns.
    /// @throws std::runtime_error naming `krhf_nuclear_forces`.
    GradientResult forces(Frame& frame);

    /// @brief Last converged total energy [Hartree].
    /// @return Total energy [Hartree], nuclear repulsion included.
    double energy() const { return energy_ha_; }

    /// @brief SCF iterations of the last converged `run`.
    /// @return Iteration count [dimensionless].
    int iterations() const { return iterations_; }

  protected:
    /// @brief The method is the energy source; no extra component is required.
    void validate_components() const override {}

  private:
    struct Loop;
    enum class Write { None, Overwrite, Accumulate };

    /// @brief Standalone SCF on this driver's Runtime.
    /// @param frame Geometry in. Energy [eV] is written when `write` says so.
    /// @param write `Overwrite` or `None`.
    void solve(Frame& frame, Write write);

    /// @brief Store the converged energy [Hartree] and write [eV] when asked.
    /// @param frame Energy destination.
    /// @param executed Iterations the driver walked.
    /// @param cap Iteration cap named in the failure [dimensionless].
    /// @param write How the total energy meets the Frame.
    /// @throws std::runtime_error when the criterion did not accept the solve.
    void commit(Frame& frame, int executed, int cap, Write write);

    std::unique_ptr<Loop> loop_;
    double energy_ha_ = 0.0;
    int iterations_ = 0;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
