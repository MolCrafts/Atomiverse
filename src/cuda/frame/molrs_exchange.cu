/// @file cuda/frame/molrs_exchange.cu
/// @brief Implementation of cuda::from_molrs / cuda::to_molrs.

#include "atomiverse/cuda/frame/molrs_exchange.cuh"

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"

namespace ATV {
namespace cuda {

Frame from_molrs(const molrs::FrameRef& src) {
    // Reuse the cpu materializer for the molrs block/column contract, then
    // transfer the host frame to the device with an explicit H2D copy.
    cpu::Frame host = cpu::from_molrs(src);
    return cpu::to_cuda(host);
}

void to_molrs(const Frame& src, molrs::FrameRef& dst) {
    // Explicit D2H into a transient host frame, then reuse cpu::to_molrs.
    cpu::Frame host(src.n_atoms());
    cuda::to_cpu(src, host);
    cpu::to_molrs(host, dst);
}

} // namespace cuda
} // namespace ATV
