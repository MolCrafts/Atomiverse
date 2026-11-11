/// @file cuda/frame/molpy_exchange.cu
/// @brief Implementation of cuda::from_molpy / cuda::to_molpy.

#include "atomiverse/cuda/frame/molpy_exchange.cuh"

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"

namespace ATV {
namespace cuda {

Frame from_molpy(const molrs::FrameRef& src) {
    // Reuse the cpu materializer for the molpy block/column contract, then
    // transfer the host frame to the device with an explicit H2D copy.
    cpu::Frame host = cpu::from_molpy(src);
    return cpu::to_cuda(host);
}

void to_molpy(const Frame& src, molrs::FrameRef& dst) {
    // Explicit D2H into a transient host frame, then reuse cpu::to_molpy.
    cpu::Frame host(src.n_atoms());
    cuda::to_cpu(src, host);
    cpu::to_molpy(host, dst);
}

} // namespace cuda
} // namespace ATV
