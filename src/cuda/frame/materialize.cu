/// @file cuda/frame/materialize.cu
/// @brief Implementation of cuda::materialize / cuda::write_back.

#include "atomiverse/cuda/frame/materialize.cuh"

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"

namespace ATV {
namespace cuda {

Frame materialize(const molrs::FrameRef& src) {
    // Reuse the cpu materializer for the molrs block/column contract, then
    // transfer the host frame to the device with an explicit H2D copy.
    cpu::Frame host = cpu::materialize(src);
    return cpu::to_cuda(host);
}

void write_back(const Frame& src, molrs::FrameRef& dst) {
    // Explicit D2H into a transient host frame, then reuse cpu::write_back.
    cpu::Frame host(src.n_atoms());
    cuda::to_cpu(src, host);
    cpu::write_back(host, dst);
}

} // namespace cuda
} // namespace ATV
