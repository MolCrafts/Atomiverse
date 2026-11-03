/// @file cpu/scf/dhf.h
/// @brief Four-component Dirac–Hartree–Fock on j-adapted spinors.
///
/// The SCF loop is `cpu::Driver<scf::Phase>` (`fock → mix → diagonalize →
/// density → check`), the same preset shape as `cpu::md::MD`. PySCF 2.14.0
/// `pyscf/scf/dhf.py` (`DHF`, `get_hcore`, `get_ovlp`, `get_jk_coulomb`,
/// `get_occ`, `energy_elec`, `kernel`) and `pyscf/scf/hf.py` (`kernel`,
/// `get_fock`, `eig`, `make_rdm1`, `energy_tot`) are that loop. It is not
/// a `MeanFieldMethod` (that class stores real matrices).
///
/// `run(Frame&)` overwrites `kEnergy` [eV]. `energy()` returns Hartree.
/// Analytic nuclear gradients are not this product — `forces` throws
/// `dhf_nuclear_forces` (`pyscf.grad.dhf` is outside this product).
#pragma once

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

/// @brief Dirac–Hartree–Fock in the kinetically balanced spinor basis.
///
/// Transplants `pyscf.scf.dhf.get_hcore`, `get_ovlp`, `get_occ`,
/// `energy_elec`, and the (LL|LL)+(SS|LL) Coulomb of `get_jk_coulomb`.
/// `run` writes `kEnergy` and `kEnergyElectronic` [eV]. `energy()` is
/// Hartree. Shells with l > 1 throw `dhf_ao_lgt1`. The (SS|SS) block
/// shifts H2 by about 1e-11 Ha and is not assembled. No Gaunt or Breit
/// term. No device path. The method does not assign convergence.
class DHF : public cpu::Driver<ATV::scf::Phase>, public HF {
  public:
    /// @brief Basis set. Fixed at construction.
    const BasisSet basis;
    /// @brief Molecular charge [e]. Electrons = ΣZ − charge.
    const int charge;
    /// @brief |ΔE| convergence threshold [Hartree], two successive steps.
    const double tol;
    /// @brief Maximum SCF iterations.
    const int max_iter;

    /// @brief Construct Dirac–Hartree–Fock.
    /// @param basis_in Gaussian basis (l ≤ 1 on this path).
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Energy change [Hartree].
    /// @param max_iter_in Iteration cap.
    DHF(BasisSet basis_in, int charge_in = 0, double tol_in = 1e-10,
        int max_iter_in = 80);

    /// @brief Release a live SCF workspace. Defined in the .cpp where the
    ///        stage loop is complete.
    ~DHF() override;

    /// @brief Move. Stage pointers address a heap loop and stay valid.
    DHF(DHF&&) noexcept;
    DHF& operator=(DHF&&) = delete;
    DHF(const DHF&) = delete;
    DHF& operator=(const DHF&) = delete;

    /// @brief This product contributes a total energy.
    bool provides_energy() const override { return true; }

    /// @brief No analytic nuclear gradient (`dhf_nuclear_forces`).
    bool provides_force() const override { return false; }

    /// @brief Whole-molecule evaluator: every atom on the rank.
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
    /// @param frame Positions [Å], species in `kAtom_atomic_number`.
    void run(Frame& frame);

    /// @brief Run SCF without writing the Frame.
    /// @param frame Geometry (read only).
    void run(const Frame& frame);

    /// @brief Analytic forces are not implemented.
    /// @param frame Unused.
    /// @return Never returns.
    /// @throws std::runtime_error naming `dhf_nuclear_forces`.
    GradientResult forces(Frame& frame);

    /// @brief Last total energy [Hartree].
    /// @return Total energy [Hartree], nuclear repulsion included.
    double energy() const { return energy_ha_; }

    /// @brief SCF iterations of the last `run`.
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
