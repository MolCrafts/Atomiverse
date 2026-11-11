#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

/// @brief Fixed-point word of a 1 eV/Angstrom force (`precision.cuh`).
/// @return Encoded word; AddOne adds it once per launch.
cuda::fixed_t unit_force_word() {
    cuda::fixed_t word = 0;
    cuda::encode_force(1.0, word);
    return word;
}

/// @brief Host image of one AddOne launch [eV/Angstrom]: forces are 64-bit
/// fixed point, so 1.0 decodes to the representable value it truncates to.
/// @return Decoded unit force.
double unit_force() { return cuda::decode_force<double>(unit_force_word()); }

__global__ void add_one_k(cuda::fixed_t* fx, cuda::fixed_t* fy,
                          cuda::fixed_t* fz, cuda::fixed_t one,
                          cuda::ForceFault* fault, const int* n_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n_ptr) {
        cuda::add_fixed(fx, idx, one, fault);
        cuda::add_fixed(fy, idx, one, fault);
        cuda::add_fixed(fz, idx, one, fault);
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
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, unit_force_word(),
            frame.force_fault, frame.atoms.d_size);
    }
};

__global__ void double_vel_k(cuda::mixed_t* vx, const int* n_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n_ptr)
        vx[idx] *= cuda::mixed_t(2);
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
    {
        auto mass = frame.span(kAtom_mass);
        for (int i = 0; i < 32; ++i)
            mass[i] = 1.0;
    }
    frame.box = cpu::Box::cubic(10.0);

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
    {
        auto fx = frame.span(kAtom_fx);
        for (int i = 0; i < 32; ++i)
            EXPECT_DOUBLE_EQ(fx[i], unit_force());
    }

    backend.shutdown();
}

TEST(Backend, BlockingDeferrable) {
    cpu::Frame frame(32);
    {
        auto mass = frame.span(kAtom_mass);
        auto vx = frame.span(kAtom_vx);
        for (int i = 0; i < 32; ++i) {
            mass[i] = 1.0;
            vx[i] = 1.0;
        }
    }
    frame.box = cpu::Box::cubic(10.0);

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
    {
        auto fx = frame.span(kAtom_fx);
        auto vx = frame.span(kAtom_vx);
        for (int i = 0; i < 32; ++i) {
            EXPECT_DOUBLE_EQ(fx[i], unit_force());
            EXPECT_DOUBLE_EQ(vx[i], 2.0);
        }
    }

    backend.shutdown();
}

TEST(Backend, ConditionalExecution) {
    cpu::Frame frame(32);
    {
        auto mass = frame.span(kAtom_mass);
        auto vx = frame.span(kAtom_vx);
        for (int i = 0; i < 32; ++i) {
            mass[i] = 1.0;
            vx[i] = 1.0;
        }
    }
    frame.box = cpu::Box::cubic(10.0);

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
    {
        auto vx = frame.span(kAtom_vx);
        for (int i = 0; i < 32; ++i)
            EXPECT_DOUBLE_EQ(vx[i], 4.0);
    }

    backend.shutdown();
}
