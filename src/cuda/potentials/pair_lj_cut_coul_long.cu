/// @file cuda/potentials/pair_lj_cut_coul_long.cu
/// @brief lj/cut/coul/long on the cluster-tile engine, plus the sparse
///        Ewald special-bond exception correction.
/// Algorithm reference: OpenMM nonbonded.cu, findInteractingBlocks.cu and
/// nonbondedExceptions.cc,
/// commit 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. Uses the published ideas
/// of parameter fusion, warp data exchange and a separate exception pass;
/// implementation is ATV-native, with no copied OpenMM source.
/// Typed LJ follows LAMMPS `lj/cut/coul/long` `init_one` semantics through
/// the shared `core/lj_type_table.h` (provenance there); each typed pair reads
/// its record from a read-only device table (LAMMPS `lib/gpu` keeps per-type
/// lj1..lj4 / cutsq / offset the same way). The exception pass applies the
/// LAMMPS `PairLJCutCoulLong::compute` cutoffs (`src/KSPACE/
/// pair_lj_cut_coul_long.cpp`, checkout `e8c4c278a8`): LJ only for
/// r² < cut_ljsq(i, j), Coulomb including the −(1 − factor_coul) qq/r
/// correction only for r² < cut_coulsq. The LJ tail correction is the LAMMPS
/// `init_one` etail/ptail (`core/lj_type_table.h`) applied as in
/// `src/compute_pe.cpp` (etail/V) and `ComputePressure::virial_compute`
/// (`src/compute_pressure.cpp`, ptail/V on the diagonal):
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "internal/cluster_tile.cuh"
#include "internal/lj_tail.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ATV::cuda {
namespace {
using ForceBox = BoxGeometry<force_t>;
using TypedTerm = tile::Sum<tile::TypedLJ, tile::Coulomb>;
using FusedTerm = tile::Sum<tile::LorentzBerthelotLJ, tile::Coulomb>;

struct Exception {
    int i, j, ab; // atom rows and table index of (type_i, type_j)
    force_t lj_scale, coul_scale;
};

// Special-bond exceptions (LAMMPS factor_lj / factor_coul with a kspace
// partner, PairLJCutCoulLong::compute): factor_lj * (LJ - offset) inside the
// pair's LJ cutoff and factor_lj * r^-4 inside its cutoff, plus, inside the
// Coulomb cutoff only, qq (factor_coul - erf(alpha r))/r — the unscaled erfc
// part minus the (1 - factor_coul) plain share the reciprocal sum added. A
// bonded pair beyond every cutoff contributes nothing, as on the CPU, whose
// neighbor list never offers it.
template <bool Energy, bool Virial>
__global__ void
exceptions(const Exception* pairs, int count, const pos_t* x, const pos_t* y,
           const pos_t* z, const force_t* q, const tile::TypedPairCoeff* table,
           Box box_arg, force_t alpha, force_t coul_cut2, fixed_t* fx,
           fixed_t* fy, fixed_t* fz, ForceFault* fault, acc_t* partial,
           int partial_stride, int partial_offset) {
    const ForceBox box = box_arg.current().as<force_t>();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t e = 0, v[6] = {};
    DeferredForceAdds<6> adds;
    if (p < count) {
        const Exception t = pairs[p];
        const int i = t.i, j = t.j;
        force_t dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        box.min_image(dx, dy, dz);
        const force_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0) {
            const force_t inv2 = 1 / r2, inv = 1 / sqrt(r2), ar = alpha / inv;
            force_t fr = 0, en = 0;
            if (t.lj_scale != 0) {
                const tile::Contribution lj =
                    tile::typed_lj(table + t.ab, r2, t.lj_scale);
                fr = lj.fpair;
                en = lj.energy;
            }
            if (r2 < coul_cut2) {
                const force_t qq = force_t(units::md_qqr2e) * q[i] * q[j] * inv;
                const force_t screened = t.coul_scale - erf(ar);
                fr += qq
                      * (screened
                         + force_t(1.1283791670955126) * ar * exp(-ar * ar))
                      * inv2;
                en += qq * screened;
            }
            const force_t f[3] = {fr * dx, fr * dy, fr * dz};
            adds.add(0, fx, i, -f[0], fault);
            adds.add(1, fy, i, -f[1], fault);
            adds.add(2, fz, i, -f[2], fault);
            adds.add(3, fx, j, f[0], fault);
            adds.add(4, fy, j, f[1], fault);
            adds.add(5, fz, j, f[2], fault);
            if constexpr (Energy)
                e = acc_t(en);
            if constexpr (Virial) {
                v[0] = acc_t(fr * dx * dx);
                v[1] = acc_t(fr * dy * dy);
                v[2] = acc_t(fr * dz * dz);
                v[3] = acc_t(fr * dx * dy);
                v[4] = acc_t(fr * dx * dz);
                v[5] = acc_t(fr * dy * dz);
            }
        }
    }
    if constexpr (Energy || Virial)
        internal::tally<Virial, kDefaultBlock>(e, v, partial, partial_stride,
                                               partial_offset + blockIdx.x);
    adds.check(fault);
}

template <class T>
void upload(T*& device, const std::vector<T>& host) {
    if (host.empty())
        return;
    ATV_CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    ATV_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
}
} // namespace

struct PairLJCutCoulLong::State {
    // Per-atom (σ/2, 2√ε) of the Lorentz–Berthelot tile; table otherwise.
    force_t* sigma = nullptr;
    force_t* epsilon = nullptr;
    tile::TypedPairCoeff* table = nullptr;
    int n_types = 0;
    bool typed = false;
    int* offsets = nullptr;
    int* partners = nullptr;
    Exception* pairs = nullptr;
    int count = 0;
    int exception_blocks = 0;
    internal::ClusterTileEngine<TypedTerm> typed_tiles;
    internal::ClusterTileEngine<FusedTerm> fused_tiles;
    ~State() {
        cudaFree(sigma);
        cudaFree(epsilon);
        cudaFree(table);
        cudaFree(offsets);
        cudaFree(partners);
        cudaFree(pairs);
    }
};
PairLJCutCoulLong::PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs,
                                     double alpha, double cutoff)
    : Pair(cutoff),
      // Refuses an empty table, a bad row and a non-positive cutoff.
      table_(std::move(coeffs), cutoff), alpha_(alpha), cut_coul_(cutoff) {
    if (!std::isfinite(alpha) || alpha <= 0)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: positive finite alpha required");
    refresh_cutoff_();
    // Same default as cpu::PairLJCutCoulLong.
    set_special_bonds_amber();
}
PairLJCutCoulLong::~PairLJCutCoulLong() = default;
PairLJCutCoulLong::PairLJCutCoulLong(PairLJCutCoulLong&&) noexcept = default;
PairLJCutCoulLong&
PairLJCutCoulLong::operator=(PairLJCutCoulLong&&) noexcept = default;

void PairLJCutCoulLong::refresh_cutoff_() {
    // LAMMPS init_one returns max(cut_lj[i][j], cut_coul); the neighbor list
    // and the tile cutoff test cover the largest of those.
    cutoff_ = std::max(cut_coul_, table_.max_cutoff());
}
PairLJCutCoulLong& PairLJCutCoulLong::set_special_bonds_amber() & {
    // LAMMPS `special_bonds amber`
    return set_special_bonds(0.0, 0.0, 0.5, 0.0, 0.0, 5.0 / 6.0);
}
PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds_amber() && {
    set_special_bonds_amber();
    return std::move(*this);
}
PairLJCutCoulLong&
PairLJCutCoulLong::set_special_bonds(double lj12, double lj13, double lj14,
                                     double c12, double c13, double c14) & {
    for (double w : {lj12, lj13, lj14, c12, c13, c14})
        if (!std::isfinite(w))
            throw std::invalid_argument(
                "cuda::PairLJCutCoulLong: special-bond weights must be finite");
    lj_[1] = lj12;
    lj_[2] = lj13;
    lj_[3] = lj14;
    coul_[1] = c12;
    coul_[2] = c13;
    coul_[3] = c14;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds(double lj12, double lj13,
                                                       double lj14, double c12,
                                                       double c13,
                                                       double c14) && {
    set_special_bonds(lj12, lj13, lj14, c12, c13, c14);
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_energy_shift(bool on) & {
    shift_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_energy_shift(bool on) && {
    shift_ = on;
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) & {
    table_.set_coeff(row);
    refresh_cutoff_();
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) && {
    set_type_coeff(row);
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_tail_correction(bool on) & {
    tail_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_tail_correction(bool on) && {
    tail_ = on;
    return std::move(*this);
}
PairLJCutCoulLong&
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) & {
    table_.set_r4(std::move(rows));
    refresh_cutoff_();
    return *this;
}
PairLJCutCoulLong
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) && {
    set_r4_attraction(std::move(rows));
    return std::move(*this);
}

void PairLJCutCoulLong::build(const BuildContext& ctx) {
    const auto& f = ctx.frame;
    if (f.n_ghost() != 0)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: single GPU required");
    state_ = std::make_unique<State>();
    auto& s = *state_;
    const int n = f.n_atoms();
    if (n > 0 && !f.atoms.type)
        throw std::invalid_argument(
            std::string("cuda::PairLJCutCoulLong: the per-type LJ table "
                        "needs the force-field type column; ")
            + atom_field_how_to("type"));
    std::vector<int> types(n);
    if (n)
        ATV_CUDA_CHECK(cudaMemcpy(types.data(), f.atoms.type, n * sizeof(int),
                                  cudaMemcpyDeviceToHost));
    int max_type = -1;
    for (int t : types) {
        if (t < 0)
            throw std::invalid_argument(
                "cuda::PairLJCutCoulLong: atom types must be >= 0");
        max_type = std::max(max_type, t);
    }
    // LAMMPS init_one over every type the Frame carries; a type no row
    // names has no LJ site.
    table_.build(max_type + 1, shift_);
    const int nt = table_.n_types();
    s.n_types = nt;
    {
        std::vector<tile::TypedPairCoeff> host(table_.pairs().size());
        for (std::size_t k = 0; k < host.size(); ++k) {
            const LJPairCoeff& c = table_.pairs()[k];
            host[k] = tile::TypedPairCoeff{
                force_t(c.lj1),  force_t(c.lj2),      force_t(c.lj3),
                force_t(c.lj4),  force_t(c.cut_ljsq), force_t(c.offset),
                force_t(c.r4_a), force_t(c.r4_cutsq), force_t(c.r4_rc5inv),
                force_t(c.r4_e0)};
        }
        upload(s.table, host);
    }
    // LAMMPS init_one tail over this build's global type counts.
    tail_pref_ = LJTail{};
    if (tail_) {
        std::vector<double> counts(static_cast<std::size_t>(nt), 0.0);
        for (int t : types)
            counts[static_cast<std::size_t>(t)] += 1.0;
        tail_pref_ = table_.tail(counts);
    }
    // The per-atom (sigma/2, 2 sqrt(eps)) tile form reproduces the table
    // only for pure Lorentz-Berthelot diagonals at one cutoff (which is the
    // Coulomb cutoff, so the shift offset is per-atom computable too).
    s.typed = !table_.lorentz_berthelot_uniform();
    typed_ = s.typed;
    if (!s.typed) {
        std::vector<force_t> sig(n), eps(n);
        for (int i = 0; i < n; ++i) {
            const LJPairCoeff& d = table_(types[i], types[i]);
            sig[i] = force_t(d.sigma * 0.5);
            eps[i] = force_t(2 * std::sqrt(d.epsilon));
        }
        upload(s.sigma, sig);
        upload(s.epsilon, eps);
    }
    // Exceptions: the nearest bond-graph shell (1, 2, 3) of a pair decides
    // its weights; only shells whose (LJ, Coulomb) weights differ from (1, 1)
    // leave the tiles. A pure (1, 1, 1, 1, 1, 1) setup has none.
    bool any_special = false;
    for (int d = 1; d <= 3; ++d)
        any_special = any_special || lj_[d] != 1.0 || coul_[d] != 1.0;
    std::vector<std::vector<int>> graph(n);
    if (any_special) {
        const int nb = f.n_bonds();
        std::vector<int> bi(nb), bj(nb);
        if (nb) {
            ATV_CUDA_CHECK(cudaMemcpy(bi.data(), f.bonds.atomi,
                                      nb * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(bj.data(), f.bonds.atomj,
                                      nb * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        }
        for (int b = 0; b < nb; ++b) {
            if (bi[b] < 0 || bj[b] < 0 || bi[b] >= n || bj[b] >= n)
                throw std::invalid_argument(
                    "cuda::PairLJCutCoulLong: bond index outside Frame");
            graph[bi[b]].push_back(bj[b]);
            graph[bj[b]].push_back(bi[b]);
        }
    }
    std::vector<int> offsets(n + 1), partners;
    std::vector<Exception> pairs;
    for (int i = 0; i < n; ++i) {
        std::map<int, int> depth;
        depth[i] = 0;
        std::vector<int> queue{i};
        for (std::size_t p = 0; p < queue.size(); ++p) {
            const int u = queue[p], d = depth[u];
            if (d == 3)
                continue;
            for (int j : graph[u])
                if (!depth.count(j)) {
                    depth[j] = d + 1;
                    queue.push_back(j);
                }
        }
        depth.erase(i);
        for (auto [j, d] : depth) {
            if (lj_[d] == 1.0 && coul_[d] == 1.0)
                continue; // an ordinary pair: stays in the tiles
            partners.push_back(j);
            if (i < j)
                pairs.push_back({i, j, types[i] * nt + types[j],
                                 force_t(lj_[d]), force_t(coul_[d])});
        }
        offsets[i + 1] = int(partners.size());
    }
    upload(s.offsets, offsets);
    upload(s.partners, partners);
    upload(s.pairs, pairs);
    s.count = int(pairs.size());
    s.exception_blocks = s.count > 0 ? grid_size(s.count) : 0;
    // Ordinary tile force excludes every exception. The sparse pass restores
    // the scaled LJ / r^-4 and the real-space Ewald exclusion correction
    // into the partial slots after the tiles.
    const potentials::CoulKernel coul{units::md_qqr2e, alpha_,
                                      cut_coul_ * cut_coul_};
    const tile::Coulomb coulomb{coul, f.atoms.charge,
                                /*masked=*/s.typed && cut_coul_ < cutoff_};
    if (s.typed)
        s.typed_tiles.build(ctx,
                            TypedTerm{{s.table, nt, f.atoms.type}, coulomb},
                            s.exception_blocks);
    else
        s.fused_tiles.build(
            ctx,
            FusedTerm{{s.sigma, s.epsilon,
                       force_t(1.0 / (cut_coul_ * cut_coul_)), shift_},
                      coulomb},
            s.exception_blocks);
    ctx.runtime.md.excl_offsets = s.offsets;
    ctx.runtime.md.excl_partners = s.partners;
    ctx.runtime.md.s_excl = 0.0;
    ctx.runtime.md.exclusions_baked = false;
}

void PairLJCutCoulLong::invalidate_inputs(InputChange changes,
                                          const BuildContext& ctx) {
    // Charges sit in the tiles' slot parameters, gathered at each rebuild.
    if (!state_ || !has_input_change(changes, InputChange::Charge))
        return;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    if (state_->typed)
        state_->typed_tiles.gather(stream);
    else
        state_->fused_tiles.gather(stream);
}

void PairLJCutCoulLong::compute(const LaunchContext& ctx) {
    const PairList* p = ctx.runtime.md.pair_list;
    if (!p || p->num_clusters == 0)
        return;
    if (p->cl_row_ptr == nullptr || p->generation == nullptr)
        throw std::runtime_error("cuda::PairLJCutCoulLong: run NeighborList "
                                 "before force evaluation");
    auto& s = *state_;
    auto& f = ctx.frame;
    if (!ctx.runtime.md.exclusions_baked)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: NeighborList must bake exception masks "
            "before force evaluation");
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    // Typed tiles test each pair's own LJ / r^-4 cutoff and the Coulomb one
    // inside the reported (largest) cutoff. The per-atom Lorentz-Berthelot
    // tile has no per-pair test: every interacting LJ pair of such a table
    // sits at the global cutoff, which is the Coulomb cutoff, so that is its
    // pair cutoff. The reported cutoff can only be larger through epsilon = 0
    // rows, which must not extend Coulomb (or any LJ pair) beyond cut_coul.
    const double pair_cut = s.typed ? cutoff_ : cut_coul_;
    const force_t cut2 = force_t(pair_cut * pair_cut);
    const force_t coul_cut2 = force_t(cut_coul_ * cut_coul_);

    auto run = [&](auto& engine, auto e, auto v) {
        constexpr bool E = decltype(e)::value, V = decltype(v)::value;
        engine.template tiles<E, V>(ctx, cut2);
        if (s.count)
            exceptions<E, V><<<s.exception_blocks, kDefaultBlock, 0, stream>>>(
                s.pairs, s.count, f.atoms.x, f.atoms.y, f.atoms.z,
                f.atoms.charge, s.table, ctx.runtime.host_box, force_t(alpha_),
                coul_cut2, f.atoms.fx, f.atoms.fy, f.atoms.fz, f.force_fault,
                engine.partial(), engine.partial_stride(),
                engine.extra_offset());
        engine.template reduce<E, V>(
            ctx, energy_term(f.scalars, EnergyKind::Pair, E),
            V ? f.scalars->stress : nullptr);
    };
    // Per-step LAMMPS eflag/vflag.
    const bool energy = ctx.energy_requested, virial = ctx.stress_requested;
    auto by_flags = [&](auto& engine) {
        using T = std::true_type;
        using F = std::false_type;
        if (energy && virial)
            run(engine, T{}, T{});
        else if (energy)
            run(engine, T{}, F{});
        else if (virial)
            run(engine, F{}, T{});
        else
            run(engine, F{}, F{});
    };
    if (s.typed)
        by_flags(s.typed_tiles);
    else
        by_flags(s.fused_tiles);
    if (tail_)
        internal::launch_lj_tail(
            ctx.runtime.host_box, tail_pref_,
            energy_term(f.scalars, EnergyKind::Pair, energy),
            virial ? f.scalars->stress : nullptr, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

int PairLJCutCoulLong::packed_tiles() const {
    if (!state_)
        return 0;
    return state_->typed ? state_->typed_tiles.packed_tiles()
                         : state_->fused_tiles.packed_tiles();
}

int PairLJCutCoulLong::valid_tiles() const {
    if (!state_)
        return 0;
    return state_->typed ? state_->typed_tiles.valid_tiles()
                         : state_->fused_tiles.valid_tiles();
}

bool PairLJCutCoulLong::typed_kernel() const { return typed_; }

void PairLJCutCoulLong::destroy(const DestroyContext& ctx) {
    // Detach the rebuild hook while the neighbor list is certainly alive.
    if (state_ && ctx.runtime.md.excl_offsets == state_->offsets) {
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
        ctx.runtime.md.exclusions_baked = false;
    }
    state_.reset();
}
} // namespace ATV::cuda
