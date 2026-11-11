#include "atomiverse/cuda/scf/mp2.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"
#include "water-chain.h"

namespace {
int run_water_chain_cuda() {
    auto frame = atv_mp2_water_chain::geometry();
    ATV::cuda::scf::RHF hf{
        ATV::cpu::scf::BasisSet::STO3G, 0, 1e-10, 200, 0.0, false};
    ATV::cuda::scf::MP2 method{hf};
    return atv_mp2_water_chain::check(method.run(frame), frame);
}
} // namespace

ATV_REGRESSION("scf/mp2", "mp2-water-chain", run_water_chain_cuda);
