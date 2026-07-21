#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

__global__ void add_one_k(cuda::real_t* fx, cuda::real_t* fy, cuda::real_t* fz,
                          const int* n_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n_ptr) {
        atomicAdd(&fx[idx], cuda::real_t(1));
        atomicAdd(&fy[idx], cuda::real_t(1));
        atomicAdd(&fz[idx], cuda::real_t(1));
    }
}

class AddOne : public cuda::Component {
  public:
    std::string name() const override { return "add_one"; }
    void launch(const cuda::LaunchContext& ctx) override {
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& frame = ctx.frame;
        int block = 256;
        int grid = (static_cast<int>(ctx.frame.n_atoms()) + block - 1) / block;
        if (grid < 1)
            grid = 1;
        add_one_k<<<grid, block, 0, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.d_size);
    }
};

__global__ void double_vel_k(cuda::real_t* vx, const int* n_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n_ptr)
        vx[idx] *= cuda::real_t(2);
}

class DoubleVel : public cuda::Component {
  public:
    std::string name() const override { return "double_vel"; }
    Scheduling scheduling() const override { return Scheduling::deferrable; }
    void launch(const cuda::LaunchContext& ctx) override {
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& frame = ctx.frame;
        int block = 256;
        int grid = (static_cast<int>(ctx.frame.n_atoms()) + block - 1) / block;
        if (grid < 1)
            grid = 1;
        double_vel_k<<<grid, block, 0, stream>>>(frame.atoms.vx,
                                                 frame.atoms.d_size);
    }
};

class FreqDoubler : public cuda::Component {
  public:
    std::string name() const override { return "freq_doubler"; }
    int frequency() const override { return 3; }
    void launch(const cuda::LaunchContext& ctx) override {
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& frame = ctx.frame;
        int block = 256;
        int grid = (static_cast<int>(ctx.frame.n_atoms()) + block - 1) / block;
        if (grid < 1)
            grid = 1;
        double_vel_k<<<grid, block, 0, stream>>>(frame.atoms.vx,
                                                 frame.atoms.d_size);
    }
};

} // anonymous namespace

TEST(Backend, KernelDispatch) {
    cpu::Frame frame(32);
    frame.atoms.mass.fill(double(1.0));
    frame.box = Mat3::cubic(10.0);

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    AddOne adder;
    PlanBuilder<cuda::Component> builder;
    builder.add({0, 0, &adder});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    adder.build(bctx);

    backend.step(0);
    backend.sync();

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    adder.destroy(dctx);

    cuda::to_cpu(backend.frame(), frame);
    for (int i = 0; i < 32; ++i)
        EXPECT_DOUBLE_EQ(frame.atoms.fx[i], 1.0);

    backend.shutdown();
}

TEST(Backend, BlockingDeferrable) {
    cpu::Frame frame(32);
    frame.atoms.mass.fill(double(1.0));
    frame.atoms.vx.fill(double(1.0));
    frame.box = Mat3::cubic(10.0);

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    AddOne adder;
    DoubleVel doubler;

    PlanBuilder<cuda::Component> builder;
    builder.add({ATV::slot_of(md::Phase::force), 0, &adder});
    builder.add({ATV::slot_of(md::Phase::end_of_step), 0, &doubler});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    adder.build(bctx);
    doubler.build(bctx);

    backend.step(0);
    backend.sync();

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    adder.destroy(dctx);
    doubler.destroy(dctx);

    cuda::to_cpu(backend.frame(), frame);
    for (int i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(frame.atoms.fx[i], 1.0);
        EXPECT_DOUBLE_EQ(frame.atoms.vx[i], 2.0);
    }

    backend.shutdown();
}

TEST(Backend, ConditionalExecution) {
    cpu::Frame frame(32);
    frame.atoms.mass.fill(double(1.0));
    frame.atoms.vx.fill(double(1.0));
    frame.box = Mat3::cubic(10.0);

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    FreqDoubler freq;
    PlanBuilder<cuda::Component> builder;
    builder.add({ATV::slot_of(md::Phase::end_of_step), 0, &freq});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    freq.build(bctx);

    backend.run(6);
    backend.sync();

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    freq.destroy(dctx);

    cuda::to_cpu(backend.frame(), frame);
    for (int i = 0; i < 32; ++i)
        EXPECT_DOUBLE_EQ(frame.atoms.vx[i], 4.0);

    backend.shutdown();
}
