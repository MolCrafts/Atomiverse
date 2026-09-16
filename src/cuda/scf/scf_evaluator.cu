/// @file cuda/scf/scf_evaluator.cu
/// @brief Synthetic-path cuda::scf::SCF only (no integral-pack ctors).

#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace ATV {
namespace cuda {
namespace scf {

SCF::SCF(int n_occ, double alpha, double tol, int max_iter)
    : cycle_(std::make_unique<internal::SyntheticScfCycle>(n_occ, alpha, tol, max_iter)) {}

void SCF::build(const BuildContext& ctx) {
    cycle_->build(ctx.runtime.stream, ctx.frame);
}

void SCF::launch(const LaunchContext& ctx) {
    cycle_->drive(ctx.runtime.stream, ctx.frame);
}

void SCF::destroy(const DestroyContext& ctx) {
    cycle_->destroy(ctx.runtime.stream, ctx.frame);
}

} // namespace scf
} // namespace cuda
} // namespace ATV
