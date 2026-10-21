/// @brief KSpacePME::launch rejects cell changes hidden by a force-box cache.
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include <gtest/gtest.h>

TEST(PmeCellGuard, RejectsCanonicalChangeEvenWithUnchangedForceBox) {
    ATV::cpu::Frame host(0);
    host.box = ATV::cpu::Box::cubic(10);
    auto device = ATV::cpu::to_cuda(host);
    ATV::cuda::Context runtime;
    for (int d = 0; d < 3; ++d)
        runtime.host_motion_box.cell[4 * d] = 10;
    runtime.host_motion_box.refresh();
    runtime.host_box =
        ATV::cuda::box_cast<ATV::cuda::real_t>(runtime.host_motion_box);
    ATV::cuda::KSpacePME pme{0.35, 1, 5};
    pme.build({runtime, device});
    runtime.host_motion_box.cell[0] +=
        sizeof(ATV::cuda::motion_t) == 8 ? 1e-8 : 1e-4;
    runtime.host_motion_box.refresh();
    EXPECT_EQ(runtime.host_box.cell[0], 10);
    EXPECT_THROW(pme.launch({runtime, device, 0, 0}), std::runtime_error);
    pme.destroy({runtime, device});
    ATV::cuda::destroy(device);
}
