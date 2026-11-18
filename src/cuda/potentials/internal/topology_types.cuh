/// @file cuda/potentials/internal/topology_types.cuh
/// @brief Build-time check that every bonded tuple's type has a row in its
///        style's coefficient table (the tables are indexed by the raw
///        topology `type_id`, as on the CPU).
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cuda/device/error.cuh"

namespace ATV::cuda::internal {

/// @brief Throw unless every one of the `count` device tuple types lies in
///        [0, n_types). Setup only (one D2H copy, allowed outside the step).
/// @param style Style name for the message.
/// @param types Device tuple-type array.
/// @param count Number of tuples.
/// @param n_types Rows in the coefficient table.
inline void require_topology_types(const char* style, const int* types,
                                   int count, int n_types) {
    if (count <= 0)
        return;
    std::vector<int> host(static_cast<std::size_t>(count));
    ATV_CUDA_CHECK(cudaMemcpy(host.data(), types, sizeof(int) * host.size(),
                              cudaMemcpyDeviceToHost));
    for (int t : host)
        if (t < 0 || t >= n_types)
            throw std::runtime_error(
                std::string(style) + ": topology type " + std::to_string(t)
                + " has no coefficients (" + std::to_string(n_types)
                + " types given; the table is indexed by the raw type_id, "
                  "so 1-based types need an entry 0)");
}

} // namespace ATV::cuda::internal
