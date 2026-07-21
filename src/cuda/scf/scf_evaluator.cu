/// @file cuda/scf/scf_evaluator.cu
/// @brief cuda::scf::SCF implementation — Evaluator component
///        wrapping the inner `SCFSolver`.
///
/// `SCF` is a thin `cuda::Component`: its lifecycle methods
/// forward to the owned `SCFSolver`. `build()` performs the one-time
/// device-workspace allocation; `launch()` runs the entire SCF
/// self-consistency cycle (the `cuda::Evaluate` outer driver makes exactly
/// one pass, so one `launch()` == one full SCF solve).

#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace ATV {
namespace cuda {
namespace scf {

SCF::SCF(int n_occ, double alpha, double tol, int max_iter)
    : solver_(std::make_unique<SCFSolver>(n_occ, alpha, tol, max_iter)) {}

#ifdef ATV_HAS_HF

SCF::SCF(const cpu::scf::IntegralPack& integrals, double tol, int max_iter,
         double level_shift)
    : solver_(
          std::make_unique<SCFSolver>(integrals, tol, max_iter, level_shift)) {}

SCF::SCF(int n_prim, const double* exponents, const double* coefficients,
         int n_occ, double tol, int max_iter)
    : solver_(std::make_unique<SCFSolver>(n_prim, exponents, coefficients,
                                          n_occ, tol, max_iter)) {}

#endif // ATV_HAS_HF

void SCF::build(const BuildContext& ctx) {
    solver_->build(ctx.runtime.stream, ctx.frame);
}

void SCF::launch(const LaunchContext& ctx) {
    solver_->run(ctx.runtime.stream, ctx.frame);
}

void SCF::destroy(const DestroyContext& ctx) {
    solver_->destroy(ctx.runtime.stream, ctx.frame);
}

} // namespace scf
} // namespace cuda
} // namespace ATV
