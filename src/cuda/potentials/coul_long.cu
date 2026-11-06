/// @file cuda/potentials/coul_long.cu
/// @brief CUDA CoulLong — real-space screened Coulomb on the cluster-tile
///        engine (`internal/cluster_tile.cuh`, term `tile::Coulomb` =
///        `potentials::CoulKernel`). Must share α with KSpacePME /
///        KSpaceEwald.
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.cuh"
#include "internal/cluster_tile.cuh"

namespace ATV {
namespace cuda {

struct CoulLong::State {
    internal::ClusterTileEngine<tile::Coulomb> tiles;
};

CoulLong::CoulLong(double alpha, double cutoff)
    : Pair(cutoff), alpha_(alpha), state_(std::make_unique<State>()) {
    if (alpha <= 0.0)
        throw std::invalid_argument("CoulLong: alpha must be > 0");
    if (cutoff <= 0.0)
        throw std::invalid_argument("CoulLong: cutoff must be > 0");
}

CoulLong::~CoulLong() = default;
CoulLong::CoulLong(CoulLong&&) noexcept = default;
CoulLong& CoulLong::operator=(CoulLong&&) noexcept = default;

void CoulLong::build(const BuildContext& ctx) {
    const potentials::CoulKernel kernel{ctx.runtime.units.qqrd2e(), alpha_,
                                        cutoff_ * cutoff_};
    state_->tiles.build(ctx, tile::Coulomb{kernel, ctx.frame.atoms.charge},
                        /*extra_partials=*/0);
}

void CoulLong::invalidate_inputs(InputChange changes, const BuildContext& ctx) {
    // Charges sit in the tiles' slot parameters, gathered at each rebuild.
    if (has_input_change(changes, InputChange::Charge))
        state_->tiles.gather(static_cast<cudaStream_t>(ctx.runtime.stream));
}

void CoulLong::destroy(const DestroyContext&) { state_->tiles.destroy(); }

void CoulLong::compute(const LaunchContext& ctx) {
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs || pairs->num_clusters <= 0)
        return;
    auto& frame = ctx.frame;
    const force_t cut2 = force_t(cutoff_) * force_t(cutoff_);
    auto run = [&](auto e, auto v) {
        constexpr bool E = decltype(e)::value, V = decltype(v)::value;
        state_->tiles.tiles<E, V>(ctx, cut2);
        state_->tiles.reduce<E, V>(
            ctx, energy_term(frame.scalars, EnergyKind::Pair, E),
            V ? &frame.scalars->stress[0] : nullptr);
    };
    // Per-step LAMMPS eflag/vflag (the step's consumers decide).
    using T = std::true_type;
    using F = std::false_type;
    const bool energy = ctx.energy_requested, virial = ctx.stress_requested;
    if (energy && virial)
        run(T{}, T{});
    else if (energy)
        run(T{}, F{});
    else if (virial)
        run(F{}, T{});
    else
        run(F{}, F{});
}

} // namespace cuda
} // namespace ATV
