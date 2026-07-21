/// @file src/cuda/potentials/bond.cu
/// @brief `cuda::Bond::select_owned_tuples` — shim until DD lands on CUDA.

#include "atomiverse/cuda/potentials/bond.cuh"

namespace ATV {
namespace cuda {

const std::vector<int>& Bond::select_owned_tuples(
    const LaunchContext& /*ctx*/, int n_tuples,
    const std::function<std::int64_t(int)>& /*owner_tag_of*/) {
    if (static_cast<int>(owned_.size()) != n_tuples) {
        owned_.resize(static_cast<std::size_t>(n_tuples));
        for (int t = 0; t < n_tuples; ++t)
            owned_[t] = t;
    }
    return owned_;
}

} // namespace cuda
} // namespace ATV
