/// @file cuda/potentials/pair_born.cu
/// @brief CUDA PairBorn — per-type Born–Mayer–Huggins on the cluster-tile
///        engine (`internal/cluster_tile.cuh`, term `tile::TypedBorn`), with
///        the LAMMPS `pair_modify tail` correction and the curvature walk
///        of the configurational temperature. Mirrors `cpu::PairBorn`.
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_born.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.cuh"
#include "internal/cluster_tile.cuh"
#include "internal/lj_tail.cuh"

namespace ATV {
namespace cuda {

namespace {

/// U(r_c) of one row without the shift.
double shift_of(const BornTypeCoeff& c, double cutoff) {
    const double rc2 = cutoff * cutoff;
    const double rc6 = rc2 * rc2 * rc2;
    return c.A * std::exp((c.sigma - cutoff) / c.rho) - c.C / rc6
           + c.D / (rc6 * rc2);
}

/// Homogeneous-tail prefactors of one ordered type pair (g(r) = 1 beyond
/// r_c): energy 2π N_i N_j ∫ r² U and virial −(2π/3) N_i N_j ∫ r³ U′ over
/// [r_c, ∞), as `cpu::PairBorn`.
LJTail born_tail(const BornTypeCoeff& c, double rc, double n_i, double n_j) {
    const double rho = c.rho, rho2 = rho * rho;
    const double rc2 = rc * rc, rc3 = rc2 * rc, rc5 = rc3 * rc2;
    const double ex = c.A * std::exp((c.sigma - rc) / rho);
    const double int_r2_u = ex * rho * (rc2 + 2.0 * rho * rc + 2.0 * rho2)
                            - c.C / (3.0 * rc3) + c.D / (5.0 * rc5);
    const double int_r3_du =
        -ex * (rc3 + 3.0 * rho * rc2 + 6.0 * rho2 * rc + 6.0 * rho2 * rho)
        + 2.0 * c.C / rc3 - 8.0 * c.D / (5.0 * rc5);
    const double pairs = n_i * n_j;
    return {2.0 * units::pi * pairs * int_r2_u,
            -(2.0 / 3.0) * units::pi * pairs * int_r3_du};
}

} // namespace

struct PairBorn::State {
    internal::ClusterTileEngine<tile::TypedBorn> tiles;
    tile::BornPairCoeff* table = nullptr; ///< Dense n_types² records.
    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() { cudaFree(table); }
};

PairBorn::PairBorn(std::vector<BornTypeCoeff> coeffs, double cutoff)
    : Pair(cutoff), coeffs_(std::move(coeffs)),
      state_(std::make_unique<State>()) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairBorn: cutoff must be > 0");
    if (coeffs_.empty())
        throw std::invalid_argument(
            "PairBorn: the table needs at least one coefficient row");
    for (const BornTypeCoeff& c : coeffs_) {
        if (c.type_i < 0 || c.type_j < 0)
            throw std::invalid_argument("PairBorn: types must be >= 0");
        if (!(c.rho > 0.0))
            throw std::invalid_argument("PairBorn: rho must be > 0");
        if (!(c.A >= 0.0))
            throw std::invalid_argument("PairBorn: A must be >= 0");
        if (!(c.C >= 0.0))
            throw std::invalid_argument("PairBorn: C must be >= 0");
    }
}

PairBorn::~PairBorn() = default;
PairBorn::PairBorn(PairBorn&&) noexcept = default;
PairBorn& PairBorn::operator=(PairBorn&&) noexcept = default;

PairBorn& PairBorn::set_energy_shift(bool on) & {
    energy_shift_ = on;
    return *this;
}
PairBorn PairBorn::set_energy_shift(bool on) && {
    energy_shift_ = on;
    return std::move(*this);
}
PairBorn& PairBorn::set_tail_correction(bool on) & {
    tail_ = on;
    return *this;
}
PairBorn PairBorn::set_tail_correction(bool on) && {
    tail_ = on;
    return std::move(*this);
}

void PairBorn::build(const BuildContext& ctx) {
    auto& fr = ctx.frame;
    if (fr.n_ghost() != 0)
        throw std::invalid_argument(
            "cuda::PairBorn needs a single GPU (no ghost rows)");
    const int rows = fr.n_atoms();
    if (rows > 0 && !fr.atoms.type)
        throw std::invalid_argument(
            std::string("cuda::PairBorn: the per-type table needs the "
                        "force-field type column; ")
            + atom_field_how_to("type"));
    std::vector<int> types(static_cast<std::size_t>(rows));
    if (rows > 0)
        ATV_CUDA_CHECK(cudaMemcpy(types.data(), fr.atoms.type,
                                  types.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
    int max_type = -1;
    for (const BornTypeCoeff& c : coeffs_)
        max_type = std::max(max_type, std::max(c.type_i, c.type_j));
    for (int t : types) {
        if (t < 0)
            throw std::invalid_argument(
                "cuda::PairBorn: atom types must be >= 0");
        max_type = std::max(max_type, t);
    }
    const int n_types = max_type + 1;
    const std::size_t nt = static_cast<std::size_t>(n_types);
    std::vector<tile::BornPairCoeff> host(nt * nt, tile::BornPairCoeff{});
    std::vector<double> counts(nt, 0.0);
    for (int t : types)
        counts[static_cast<std::size_t>(t)] += 1.0;
    for (const BornTypeCoeff& c : coeffs_) {
        const tile::BornPairCoeff e{
            force_t(c.A),
            force_t(1.0 / c.rho),
            force_t(c.sigma),
            force_t(c.C),
            force_t(c.D),
            force_t(energy_shift_ ? shift_of(c, cutoff_) : 0.0),
            1};
        host[static_cast<std::size_t>(c.type_i) * nt + c.type_j] = e;
        host[static_cast<std::size_t>(c.type_j) * nt + c.type_i] = e;
    }
    // LAMMPS init_one tail over ordered type pairs with the type counts of
    // this build; a later row for the same pair replaced the earlier one.
    etail_pref_ = ptail_pref_ = 0.0;
    if (tail_) {
        std::vector<const BornTypeCoeff*> row(nt * nt, nullptr);
        for (const BornTypeCoeff& c : coeffs_) {
            row[static_cast<std::size_t>(c.type_i) * nt + c.type_j] = &c;
            row[static_cast<std::size_t>(c.type_j) * nt + c.type_i] = &c;
        }
        for (std::size_t ti = 0; ti < nt; ++ti)
            for (std::size_t tj = 0; tj < nt; ++tj) {
                if (!row[ti * nt + tj])
                    continue;
                const LJTail t = born_tail(*row[ti * nt + tj], cutoff_,
                                           counts[ti], counts[tj]);
                etail_pref_ += t.energy;
                ptail_pref_ += t.virial;
            }
    }
    cudaFree(state_->table);
    state_->table = nullptr;
    ATV_CUDA_CHECK(
        cudaMalloc(&state_->table, host.size() * sizeof(tile::BornPairCoeff)));
    ATV_CUDA_CHECK(cudaMemcpy(state_->table, host.data(),
                              host.size() * sizeof(tile::BornPairCoeff),
                              cudaMemcpyHostToDevice));
    state_->tiles.build(ctx,
                        tile::TypedBorn{state_->table, n_types, fr.atoms.type},
                        /*extra_partials=*/0);
}

void PairBorn::destroy(const DestroyContext&) {
    state_->tiles.destroy();
    cudaFree(state_->table);
    state_->table = nullptr;
}

void PairBorn::compute(const LaunchContext& ctx) {
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs || pairs->num_clusters <= 0)
        return;
    auto& frame = ctx.frame;
    const force_t cut2 = force_t(cutoff_) * force_t(cutoff_);
    const bool energy = ctx.energy_requested, virial = ctx.stress_requested;
    acc_t* const stress = virial ? &frame.scalars->stress[0] : nullptr;
    auto run = [&](auto e, auto v) {
        constexpr bool E = decltype(e)::value, V = decltype(v)::value;
        state_->tiles.tiles<E, V>(ctx, cut2);
        state_->tiles.reduce<E, V>(
            ctx, energy_term(frame.scalars, EnergyKind::Pair, E), stress);
    };
    // Per-step LAMMPS eflag/vflag (the step's consumers decide).
    using T = std::true_type;
    using F = std::false_type;
    if (energy && virial)
        run(T{}, T{});
    else if (energy)
        run(T{}, F{});
    else if (virial)
        run(F{}, T{});
    else
        run(F{}, F{});
    if (tail_)
        internal::launch_lj_tail(
            ctx.runtime.host_box, LJTail{etail_pref_, ptail_pref_},
            energy_term(frame.scalars, EnergyKind::Pair, energy), stress,
            static_cast<cudaStream_t>(ctx.runtime.stream));
}

void PairBorn::add_curvature(const LaunchContext& ctx,
                             const CurvatureRows& rows) {
    state_->tiles.curvature(ctx, force_t(cutoff_) * force_t(cutoff_), rows);
}

} // namespace cuda
} // namespace ATV
