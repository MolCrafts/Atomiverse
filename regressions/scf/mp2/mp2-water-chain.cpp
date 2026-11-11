#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"
#include "water-chain.h"

namespace {
int run_water_chain() {
    using namespace ATV::cpu::scf;
    auto frame = atv_mp2_water_chain::geometry();
    RHF hf{BasisSet::STO3G, 0, 1e-10, 200};
    hf.init_guess = InitGuess::Hcore;
    hf.diis_start_iter = 1;
    hf.diis_max_vectors = 8;
    MP2<RHF> method{hf};
    return atv_mp2_water_chain::check(method.run(frame), frame);
}
} // namespace

ATV_REGRESSION("scf/mp2", "mp2-water-chain", run_water_chain);
