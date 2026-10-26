/// @file cuda/scf/uhf.cuh
/// @brief cuda::scf::UHF — unrestricted Hartree–Fock on the CUDA backend.
///
/// Mirror of `cpu::scf::UHF` for the device envelope: constructor arguments,
/// Frame in/out, SCF as `cuda::Driver<scf::Phase>`
/// (`fock → mix → diagonalize → density → check`).
///
/// **Device only.** STO-3G with every shell l ≤ 1 builds S, T, V, ERI, both
/// Fock matrices, both densities, CDIIS, and the analytic gradient on device.
/// Anything else throws and names the missing kernel (`launch_eri_sp` /
/// `launch_uhf_forces_sp`). There is no host-ERI fill.
///
/// PySCF sources read for this type: `scf/uhf.py` (`make_rdm1`, `get_veff`,
/// `energy_elec`, `get_fock`, `get_occ`, `init_guess_by_minao`), `scf/hf.py`
/// (`init_guess_by_minao`, `kernel`), `grad/uhf.py` (`grad_elec`, `get_veff`,
/// `make_rdm1e`). The device loop starts from the spin-averaged MINAO density
/// (open-shell `_break_dm_spin_symm` is a no-op). Overlap projection of that
/// guess uses device overlap integrals and a device Cholesky solve. Only
/// geometry-independent element constants are uploaded at setup. An element
/// without a MINAO table keeps the core-Hamiltonian start.
/// `scf/dhf.py`, `pbc/scf/khf.py`, `x2c/sfx2c1e_grad.py`, `df/df_jk.py`, and
/// `hessian/rhf.py` are different products and are not members.
///
/// ΔSCF / MOM stays on `cpu::scf::UHF::delta_scf`. `Hessian<UHF>`,
/// `SfTDA<UHF>`, and `CCSD<UHF>` are not members.
///
/// Units: `kEnergy` [eV]; forces `kAtom_f*` [eV/Å]; `energy()` [Hartree].
///
/// @code{.cpp}
///   cuda::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/1,
///                      /*n_beta=*/0, /*charge=*/1};
///   uhf.run(frame);
///   uhf.forces(frame);
/// @endcode
#pragma once

#include <memory>
#include <stdexcept>
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

struct UhfLoopState;

/// @brief Unrestricted Hartree–Fock on the CUDA backend.
class UHF : public cuda::Driver<ATV::scf::Phase>,
            public cuda::DriverNode,
            public cuda::HF {
  public:
    /// @brief Basis set (set at construction only).
    const cpu::scf::BasisSet basis;
    /// @brief Alpha electrons (occupation 1 on the lowest α eigenvectors).
    const int n_alpha;
    /// @brief Beta electrons (occupation 1 on the lowest β eigenvectors).
    const int n_beta;
    /// @brief Molecular charge [e]. `n_alpha + n_beta` must equal ΣZ − charge.
    const int charge;
    /// @brief Density residual Frobenius tolerance [dimensionless].
    const double tol;
    /// @brief Maximum SCF iterations [dimensionless].
    const int max_iter;
    /// @brief Virtual-orbital level shift λ [Hartree], both spins (0 = off).
    const double level_shift;

    /// @brief Nested Evaluate may assemble forces after SCF. Mutable only
    ///        for `with_gradient`. Standalone `forces()` always computes them.
    bool compute_force = true;

    /// @brief Construct UHF parameters.
    /// @param basis_in Basis set. Device path accepts STO-3G only.
    /// @param n_alpha_in Alpha electrons.
    /// @param n_beta_in Beta electrons.
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Residual tolerance [dimensionless].
    /// @param max_iter_in Max SCF iterations.
    /// @param level_shift_in Virtual shift [Hartree], applied per spin.
    /// @param compute_force_in Nested force assembly default.
    explicit UHF(cpu::scf::BasisSet basis_in, int n_alpha_in, int n_beta_in,
                 int charge_in = 0, double tol_in = 1e-9, int max_iter_in = 100,
                 double level_shift_in = 0.0, bool compute_force_in = true);

    ~UHF() override;
    UHF(UHF&&) noexcept;
    UHF& operator=(UHF&&) = delete;
    UHF(const UHF&) = delete;
    UHF& operator=(const UHF&) = delete;

    /// @brief Enable or disable nested force assembly.
    UHF& with_gradient(bool enabled) & noexcept {
        compute_force = enabled;
        return *this;
    }
    /// @brief Enable or disable nested force assembly (rvalue).
    UHF&& with_gradient(bool enabled) && noexcept {
        compute_force = enabled;
        return std::move(*this);
    }

    bool provides_energy() const override { return true; }
    bool provides_force() const override { return compute_force; }
    int default_budget() const override { return max_iter; }

    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

    /// @brief One-shot SCF. Writes `kEnergy` [eV].
    /// @param frame Geometry [Å, Z].
    void run(cpu::Frame& frame);

    /// @brief SCF plus analytic nuclear forces.
    /// @param frame Geometry [Å, Z]. Forces [eV/Å] and energy [eV] written.
    /// @return Forces [eV/Å] and energy [eV].
    cpu::scf::GradientResult forces(cpu::Frame& frame);

    /// @brief Total energy of the last successful standalone solve [Hartree].
    /// @throws std::logic_error Before standalone success, after failure, or
    /// after a nested solve; nested results reside on the parent's Frame.
    double energy() const {
        if (!energy_available_)
            throw std::logic_error(
                "cuda::scf::UHF::energy: no successful standalone result; "
                "nested energy is on the parent Frame");
        return energy_ha_;
    }
    /// @brief SCF iterations of the most recent solve.
    int iterations() const { return iterations_; }
    /// @brief True when the last force assembly used `launch_uhf_forces_sp`.
    bool last_force_on_device() const { return last_force_device_; }
    /// @brief True when the last SCF used device s/p integrals.
    bool last_scf_on_device() const { return last_device_sp_; }

  protected:
    void validate_components() const override {}

  private:
    void cache_host_types(const cpu::Frame& frame);
    void wire_stages_();
    int drive_loop_(cuda::Context& ctx, cuda::Frame& frame, int budget,
                    bool host_current = false);
    void step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step);
    void assemble_(cuda::Context& ctx, cuda::Frame& frame, bool host_current);
    void release_engine_();
    void assemble_forces_(cuda::Context& ctx, cuda::Frame& frame);
    void pull_energy_(cuda::Context& ctx, cuda::Frame& frame);

    std::unique_ptr<UhfLoopState> loop_;
    double energy_ha_ = 0.0;
    bool energy_available_ = false;
    int iterations_ = 0;
    bool last_force_device_ = false;
    bool last_device_sp_ = false;
    std::vector<int> host_Z_;
};

} // namespace scf
} // namespace cuda
} // namespace ATV
