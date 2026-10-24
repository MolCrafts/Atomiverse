/// @file cuda/semiempirical/rm1_solver.cuh
/// @brief Native CUDA RM1 self-consistency Driver, energy, charges and analytic forces.
#pragma once
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"
#include "atomiverse/semiempirical/nddo_result.h"
#include <memory>

namespace ATV::cuda::semiempirical {
namespace internal {
struct NddoWorkspace;
}
using RM1Result = ATV::semiempirical::NDDOResult;

/// FP64 neutral, closed-shell CHNO molecules without periodic cells.
/// The concrete method owns diagonalize → density → fock → check and its
/// Criterion. Device matrices and analytic-force scratch persist across solves.
class RM1 : public cuda::Driver<ATV::nddo::Phase>, public cuda::DriverNode {
  public:
    const int charge, multiplicity;
    int max_iter;
    double density_tol, energy_tol;
    explicit RM1(int charge = 0, int multiplicity = 1, int max_iter = 200,
                 double density_tol = 1e-8, double energy_tol = 1e-10);
    ~RM1();
    RM1(RM1&& other);
    RM1(const RM1& other);
    RM1& operator=(RM1&&) = delete;
    RM1& operator=(const RM1&) = delete;
    using cuda::Driver<ATV::nddo::Phase>::build;
    /// Prepare persistent device workspace for the supplied molecular identity.
    void build(const cpu::Frame& frame);
    /// Compute energy/charges, overwrite these outputs, preserve existing forces.
    RM1Result& run(cpu::Frame& frame);
    /// Compute method results without modifying any caller Frame field.
    RM1Result& run(const cpu::Frame& frame);
    /// Compute energy, charges and analytic forces; overwrite these Frame outputs.
    RM1Result& forces(cpu::Frame& frame);
    /// Explicit result-read boundary; downloads charges and requested forces.
    const RM1Result& last_result() const;
    bool has_density() const noexcept;
    bool built() const noexcept;
    std::string name() const { return "rm1"; }
    bool provides_energy() const { return true; }
    bool provides_force() const { return true; }
    bool provides_system_stress() const { return false; }
    int default_budget() const override { return -1; }
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;
    void shutdown() override;

  protected:
    void validate_components() const override {}

  private:
    void start_();
    void standalone_(const cpu::Frame& frame);
    RM1Result& run_host_(const cpu::Frame& frame, bool force_request);
    int solve_(cuda::Context& ctx, cuda::Frame& frame, int budget,
               bool accumulate);
    void release_backend_();
    std::unique_ptr<internal::NddoWorkspace> workspace_;
    cuda::Context* attached_ = nullptr;
    void* last_stream_ = nullptr;
    bool active_ = false;
    int standalone_atoms_ = 0;
};
} // namespace ATV::cuda::semiempirical
