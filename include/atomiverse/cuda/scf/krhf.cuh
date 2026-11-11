/// @file cuda/scf/krhf.cuh
/// @brief cuda::scf::KRHF — k-point restricted Hartree–Fock on device.
///
/// Same preset shape as `cpu::scf::KRHF`: `cuda::Driver<scf::Phase>`
/// (`fock → mix → diagonalize → density → check`) plus the `cuda::HF` tag.
/// Setup packs geometry, the s-basis, the FFT mesh and the core Hamiltonian.
/// The SCF step (Coulomb, exchange, Fock, `zheevd`, occupation, density)
/// runs on device. l > 0 throws `krhf_ao_lgt0`. Forces throw
/// `krhf_nuclear_forces`.
#pragma once

#include <array>
#include <memory>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"

namespace ATV {
namespace cuda {
namespace scf {

struct KrhfLoop;

/// @brief Device KRHF (FFTDF J/K, Ewald exchange divergence).
class KRHF : public cuda::Driver<ATV::scf::Phase>,
             public cuda::DriverNode,
             public cuda::HF {
  public:
    /// @brief Basis set. Fixed at construction.
    const cpu::scf::BasisSet basis;
    /// @brief Monkhorst–Pack divisions. Each entry ≥ 1.
    const std::array<int, 3> k_mesh;
    /// @brief Cell charge [e].
    const int charge;
    /// @brief |ΔE| threshold [Hartree].
    const double tol;
    /// @brief Iteration cap.
    const int max_iter;

    /// @brief Construct device k-point RHF.
    /// @param basis_in Gaussian basis (s functions on this path).
    /// @param k_mesh_in Monkhorst–Pack mesh.
    /// @param charge_in Cell charge [e].
    /// @param tol_in Energy change [Hartree].
    /// @param max_iter_in Iteration cap.
    KRHF(cpu::scf::BasisSet basis_in, std::array<int, 3> k_mesh_in,
         int charge_in = 0, double tol_in = 1e-9, int max_iter_in = 40);

    /// @brief Release a live device workspace. Defined in the .cu.
    ~KRHF() override;
    /// @brief Move. Stage pointers address a heap loop and stay valid.
    KRHF(KRHF&&) noexcept;
    KRHF& operator=(KRHF&&) = delete;
    KRHF(const KRHF&) = delete;
    KRHF& operator=(const KRHF&) = delete;

    /// @brief This product contributes a total energy.
    bool provides_energy() const override { return true; }
    /// @brief No analytic nuclear gradient on this path.
    bool provides_force() const override { return false; }

    /// @brief Nested budget is the SCF iteration cap.
    /// @return `max_iter` [dimensionless].
    int default_budget() const override { return max_iter; }

    /// @brief Compile the stage Plan. Device buffers wait until the solve.
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;

    /// @brief One SCF on the parent's device frame. Accumulates the energy
    ///        scalar [eV].
    /// @param ctx Parent context (stream + loop slot, restored on exit).
    /// @param frame Parent device frame.
    /// @param budget SCF iterations; `<= 0` uses `max_iter`.
    /// @return Iterations executed.
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;

    /// @brief Drop a device workspace left open by an interrupted nest.
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

    /// @brief SCF. Overwrites `kEnergy` and `kEnergyElectronic` [eV].
    /// @param frame Orthorhombic cell, positions [Å].
    void run(cpu::Frame& frame);

    /// @brief SCF without writing the Frame.
    /// @param frame Orthorhombic cell (read only).
    void run(const cpu::Frame& frame);

    /// @brief Throws `krhf_nuclear_forces`.
    /// @param frame Unused.
    /// @return Never returns.
    cpu::scf::GradientResult forces(cpu::Frame& frame);

    /// @brief Last total energy [Hartree].
    /// @return Total energy [Hartree], nuclear repulsion included.
    double energy() const { return energy_ha_; }
    /// @brief Iterations of the last converged `run`.
    /// @return Iteration count [dimensionless].
    int iterations() const { return iterations_; }

  protected:
    /// @brief The method is the energy source; no extra component is required.
    void validate_components() const override {}

  private:
    enum class Write { None, Overwrite, Accumulate };

    /// @brief SCF on `host` geometry, kernels on `ctx.stream`.
    /// @param ctx Stream and loop slot.
    /// @param host Geometry the assembler reads.
    /// @param device Parent device frame when accumulating; null standalone.
    /// @param budget Iteration cap; `<= 0` uses `max_iter`.
    /// @param write Host overwrite, device accumulate, or no Frame write.
    /// @return Iterations executed.
    int drive(cuda::Context& ctx, cpu::Frame& host, cuda::Frame* device,
              int budget, Write write);

    /// @brief Walk this Driver's Plan once (one SCF iteration).
    void step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step);

    /// @brief Sync, destroy components, shut the backend down. Does not
    ///        retrieve: `to_cpu` would overwrite host meta from the device
    ///        scalar, which this SCF does not publish.
    void release_backend_();

    std::unique_ptr<KrhfLoop> loop_;
    double energy_ha_ = 0.0;
    int iterations_ = 0;
};

} // namespace scf
} // namespace cuda
} // namespace ATV
