/// @file cpu/scf/internal/exact_jk.cpp
#include "atomiverse/cpu/scf/internal/exact_jk.h"

#include "exact_jk_contraction.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/rys_eri.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace ATV::cpu::scf::internal {
namespace {
using exact_jk_detail::contract_j_n;
using exact_jk_detail::contract_k_n;
using exact_jk_detail::pair_index;

double resident_mb() {
#if defined(__linux__)
    std::ifstream status("/proc/self/statm");
    std::size_t total = 0, resident = 0;
    if (status >> total >> resident)
        return static_cast<double>(resident) * sysconf(_SC_PAGESIZE) / 1e6;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count)
        == KERN_SUCCESS)
        return static_cast<double>(info.resident_size) / 1e6;
#endif
    return 0;
}
// Visit each symmetry-unique AO quartet once. Block-pair order is not AO-pair
// order, so only a diagonal block quartet needs the ab >= cd filter.
// `bound(bra, ket)` scales the Schwarz product by the largest density element
// the quartet can touch (1 for cache construction).
template <typename Consume, typename Bound>
void visit_quartets(const ShellBlockPairList& pairs, int left,
                    Consume&& consume, Bound&& bound, double omega,
                    const std::vector<RealMatrix>* transforms = nullptr,
                    const std::vector<int>* starts = nullptr) {
    const auto& bra = pairs.pairs[left];
    const auto& A = pairs.blocks[bra.si];
    const auto& B = pairs.blocks[bra.sj];
    double block[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                 * kMaxNCartOrbital];
    for (int right = 0; right <= left; ++right) {
        const auto& ket = pairs.pairs[right];
        // erf(ω r)/r ≤ 1/r in Fourier space, so Coulomb Schwarz bounds the
        // attenuated quartets too.
        if (bra.schwarz * ket.schwarz * bound(bra, ket) < pairs.threshold)
            continue;
        const auto& C = pairs.blocks[ket.si];
        const auto& D = pairs.blocks[ket.sj];
        const int size = A.n_cart * B.n_cart * C.n_cart * D.n_cart;
        std::fill_n(block, size, 0.0);
        if (omega > 0)
            native::rys_eri_erf_block(A, B, C, D, omega, block);
        else
            native::eri_block(A, B, C, D, block);
        std::array<int, 4> dims{A.n_cart, B.n_cart, C.n_cart, D.n_cart};
        std::array<int, 4> offsets{A.ao_start, B.ao_start, C.ao_start,
                                   D.ao_start};
        if (transforms) {
            // Transform one shell quartet, never a dense n^4 intermediate.
            // Each axis is a small contraction with the existing basis map.
            double scratch[kMaxNCartOrbital * kMaxNCartOrbital
                           * kMaxNCartOrbital * kMaxNCartOrbital];
            const int shells[] = {bra.si, bra.sj, ket.si, ket.sj};
            for (int axis = 0; axis < 4; ++axis) {
                const auto& T = (*transforms)[shells[axis]];
                const int old_dim = dims[axis], new_dim = T.shape()[1];
                int before = 1, after = 1;
                for (int i = 0; i < axis; ++i)
                    before *= dims[i];
                for (int i = axis + 1; i < 4; ++i)
                    after *= dims[i];
                for (int i = 0; i < before; ++i)
                    for (int s = 0; s < new_dim; ++s)
                        for (int j = 0; j < after; ++j) {
                            double v = 0;
                            for (int c = 0; c < old_dim; ++c)
                                v += T(c, s)
                                     * block[(i * old_dim + c) * after + j];
                            scratch[(i * new_dim + s) * after + j] = v;
                        }
                std::copy_n(scratch, before * new_dim * after, block);
                dims[axis] = new_dim;
                offsets[axis] = (*starts)[shells[axis]];
            }
        }
        for (int ia = 0; ia < dims[0]; ++ia) {
            const int a = offsets[0] + ia;
            for (int ib = 0; ib < dims[1]; ++ib) {
                const int b = offsets[1] + ib;
                if (a < b)
                    continue;
                const auto ab = pair_index(a, b);
                for (int ic = 0; ic < dims[2]; ++ic) {
                    const int c = offsets[2] + ic;
                    for (int id = 0; id < dims[3]; ++id) {
                        const int d = offsets[3] + id;
                        if (c < d || (left == right && ab < pair_index(c, d)))
                            continue;
                        const double v =
                            block[((ia * dims[1] + ib) * dims[2] + ic) * dims[3]
                                  + id];
                        consume(a, b, c, d, v);
                    }
                }
            }
        }
    }
}

// Compile-time density counts for the common cases (RHF/KS: 1, UHF: 2,
// GHF: 4 exchange blocks), runtime loop otherwise.
inline void contract_j(int n, int m, const double* p, double* j, int a, int b,
                       int c, int d, double v) {
    switch (m) {
    case 1:
        return contract_j_n<1>(n, m, p, j, a, b, c, d, v);
    case 2:
        return contract_j_n<2>(n, m, p, j, a, b, c, d, v);
    default:
        return contract_j_n<0>(n, m, p, j, a, b, c, d, v);
    }
}

inline void contract_k(int n, int m, const double* p, double* k,
                       const double* scale, int a, int b, int c, int d,
                       double v) {
    switch (m) {
    case 1:
        return contract_k_n<1>(n, m, p, k, scale, a, b, c, d, v);
    case 2:
        return contract_k_n<2>(n, m, p, k, scale, a, b, c, d, v);
    case 4:
        return contract_k_n<4>(n, m, p, k, scale, a, b, c, d, v);
    default:
        return contract_k_n<0>(n, m, p, k, scale, a, b, c, d, v);
    }
}

// Largest |P| over each shell-block pair (both orientations), maximized over
// densities — the density factor of a quartet's screening bound.
void block_density_max(const ShellBlockPairList& pairs,
                       const std::vector<const double*>& densities, int n,
                       std::vector<double>& out) {
    const int nb = static_cast<int>(pairs.blocks.size());
    out.assign(static_cast<std::size_t>(nb) * nb, 0.0);
    for (int x = 0; x < nb; ++x) {
        const auto& X = pairs.blocks[x];
        for (int y = 0; y <= x; ++y) {
            const auto& Y = pairs.blocks[y];
            double m = 0;
            for (const double* p : densities)
                for (int i = X.ao_start; i < X.ao_start + X.n_cart; ++i)
                    for (int j = Y.ao_start; j < Y.ao_start + Y.n_cart; ++j)
                        m = std::max(m, std::max(std::abs(p[i + j * n]),
                                                 std::abs(p[j + i * n])));
            out[static_cast<std::size_t>(x) * nb + y] = m;
            out[static_cast<std::size_t>(y) * nb + x] = m;
        }
    }
}
} // namespace

void ExactJK::prepare(const IntegralView& ints, double memory_mb, bool mpi,
                      double omega) {
    if (!std::isfinite(memory_mb) || memory_mb < 0)
        throw std::invalid_argument(
            "max_memory_mb must be finite and nonnegative");
    if (!std::isfinite(omega) || omega < 0)
        throw std::invalid_argument(
            "ExactJK: range-separation omega must be finite and >= 0");
    const auto start = LogClock::now();
    history_.clear();
    omega_ = omega;
    rank_ = 0;
    ranks_ = 1;
    if (mpi) {
#ifdef ATV_HAS_MPI
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        if (!initialized || finalized)
            throw std::runtime_error(
                "HF mpi=true requires an active MPI runtime on every rank");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks_);
#else
        throw std::runtime_error("HF mpi=true requires ATV_USE_MPI=ON");
#endif
    }
    // Release the previous geometry before deciding/allocating the next cache.
    std::vector<double>().swap(eri_);
    pairs_ = {};
    c2s_ = {};
    // Explicit dense inputs are used by periodic/post-HF callers and synthetic
    // unit fixtures. Compress them; never select the old dense J/K algorithm.
    const bool dense = ints.ERI.size() != 0 && ints.shells.empty();
    n_ = ints.n_basis;
    if (n_ <= 0)
        throw std::invalid_argument("ExactJK requires a nonempty AO basis");
    if (dense && omega_ > 0)
        throw std::invalid_argument(
            "ExactJK: erf-attenuated exchange needs shells, not dense ERIs");
    const bool from_dense = ints.ERI.size() != 0 && omega_ == 0;
    const auto n = static_cast<std::size_t>(n_);
    const auto np = n * (n + 1) / 2;
    if (np > std::numeric_limits<std::size_t>::max() / (np + 1))
        throw std::overflow_error("ExactJK s8 size overflow");
    const auto count = np * (np + 1) / 2;
    if (!from_dense) {
        if (ints.shells.empty())
            throw std::invalid_argument("ExactJK requires shells or ERIs");
        // Match PySCF's default direct-SCF integral threshold.
        pairs_ = ShellBlockPairList(ints, 1e-13);
    }
    // PySCF RHF._is_mem_enough: nbf**4 / 1e6 + current_memory < .95 * max_memory.
    // Use actual s8 bytes as well, since n**4 underestimates tiny s8 arrays.
    const double estimate =
        std::max(std::pow(static_cast<double>(n), 4),
                 static_cast<double>(count) * sizeof(double));
    int cache = from_dense || estimate / 1e6 + resident_mb() < .95 * memory_mb;
#ifdef ATV_HAS_MPI
    if (ranks_ > 1)
        MPI_Allreduce(MPI_IN_PLACE, &cache, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
    if (cache) {
        eri_.assign(count, 0.0);
        if (from_dense) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int a = 0; a < n_; ++a)
                for (int b = 0; b <= a; ++b) {
                    const auto ab = pair_index(a, b);
                    for (int c = 0; c <= a; ++c)
                        for (int d = 0; d <= c; ++d) {
                            const auto cd = pair_index(c, d);
                            if (cd <= ab)
                                eri_[pair_index(ab, cd)] = ints.ERI(a, b, c, d);
                        }
                }
        } else {
            std::vector<RealMatrix> transforms;
            std::vector<int> starts;
            if (ints.spherical()) {
                int offset = 0;
                for (const auto& block : pairs_.blocks) {
                    const int ns = 2 * block.l + 1;
                    RealMatrix T = xt::zeros<double>({block.n_cart, ns});
                    for (int i = 0; i < block.n_cart; ++i)
                        for (int j = 0; j < ns; ++j)
                            T(i, j) = ints.c2s(block.ao_start + i, offset + j);
                    transforms.push_back(std::move(T));
                    starts.push_back(offset);
                    offset += ns;
                }
            }
            const auto unit = [](const ShellBlockPair&, const ShellBlockPair&) {
                return 1.0;
            };
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int left = 0; left < static_cast<int>(pairs_.pairs.size());
                 ++left)
                visit_quartets(
                    pairs_, left,
                    [&](int a, int b, int c, int d, double v) {
                        eri_[pair_index(pair_index(a, b), pair_index(c, d))] =
                            v;
                    },
                    unit, omega_, transforms.empty() ? nullptr : &transforms,
                    transforms.empty() ? nullptr : &starts);
        }
    } else {
        n_ = ints.n_ao_cart();
        if (ints.spherical())
            c2s_ = ints.c2s;
    }
    ATV_INFO("[exact_jk] cache_bytes={} build_ms={:.3f} mpi_ranks={} omega={}",
             cache_bytes(), elapsed_ms(start), ranks_, omega_);
}

std::optional<double> ExactJK::coulomb_cache_screening_bound() const {
    if (!cached() || omega_ != 0.0)
        return std::nullopt;
    if (pairs_.blocks.empty())
        return 0.0; // Cache filled from a supplied dense tensor.
    // Whole pairs are discarded at Q <= threshold. For retained pairs the
    // quartet test drops Q_bra * Q_ket < threshold.
    double max_schwarz = std::max(1.0, pairs_.threshold);
    for (const auto& pair : pairs_.pairs)
        max_schwarz = std::max(max_schwarz, pair.schwarz);
    return pairs_.threshold * max_schwarz;
}

void ExactJK::unpack_coulomb_cache_row(std::size_t pair, double* ao) const {
    const auto n = static_cast<std::size_t>(n_);
    std::size_t other = 0;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b <= a; ++b, ++other) {
            const double value = eri_[pair_index(pair, other)];
            ao[a * n + b] = value;
            ao[b * n + a] = value;
        }
}

void ExactJK::compute(std::span<const JKRequest> requests) const {
    const auto start = LogClock::now();
    if (n_ <= 0)
        throw std::logic_error("ExactJK::compute before prepare");
    const std::size_t nao = c2s_.size() ? c2s_.shape()[1] : n_;
    const std::size_t nr = requests.size();
    // Engine-basis densities (Cartesian on the streamed spherical route).
    std::vector<RealMatrix> engine(nr);
    std::vector<const double*> j_densities, k_densities;
    for (std::size_t r = 0; r < nr; ++r) {
        const JKRequest& q = requests[r];
        if (!q.P)
            throw std::invalid_argument("ExactJK request without density");
        if (q.P->shape()[0] != nao || q.P->shape()[1] != nao)
            throw std::invalid_argument("ExactJK density dimension mismatch");
        if (c2s_.size())
            engine[r] = xt::linalg::dot(
                c2s_, RealMatrix(xt::linalg::dot(*q.P, xt::transpose(c2s_))));
        if (q.J)
            j_densities.push_back(c2s_.size() ? engine[r].data() : q.P->data());
        if (q.K)
            k_densities.push_back(c2s_.size() ? engine[r].data() : q.P->data());
    }
    const auto density = [&](std::size_t r) {
        return c2s_.size() ? engine[r].data() : requests[r].P->data();
    };
    // Per-block-pair density bounds: J touches P_ab and P_cd, K touches
    // P_ac, P_ad, P_bc and P_bd of the quartet (ab|cd).
    std::vector<double> dj, dk;
    int nb = 0;
    if (!cached()) {
        nb = static_cast<int>(pairs_.blocks.size());
        if (!j_densities.empty())
            block_density_max(pairs_, j_densities, n_, dj);
        if (!k_densities.empty())
            block_density_max(pairs_, k_densities, n_, dk);
    }
    const auto bound = [&](const ShellBlockPair& bra,
                           const ShellBlockPair& ket) {
        const auto at = [&](const std::vector<double>& m, int x, int y) {
            return m[static_cast<std::size_t>(x) * nb + y];
        };
        double b = 0;
        if (!dj.empty())
            b = std::max(at(dj, bra.si, bra.sj), at(dj, ket.si, ket.sj));
        if (!dk.empty())
            b = std::max({b, at(dk, bra.si, ket.si), at(dk, bra.si, ket.sj),
                          at(dk, bra.sj, ket.si), at(dk, bra.sj, ket.sj)});
        return b;
    };

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    const std::size_t nn = static_cast<std::size_t>(n_) * n_;
    // Interleaved J and K densities (element-major) and their requests.
    std::vector<std::size_t> jr, kr;
    for (std::size_t r = 0; r < nr; ++r) {
        if (requests[r].J)
            jr.push_back(r);
        if (requests[r].K)
            kr.push_back(r);
    }
    const int mj = static_cast<int>(jr.size()),
              mk = static_cast<int>(kr.size());
    std::vector<double> dj_in(nn * mj), dk_in(nn * mk), scale(mk);
    for (int t = 0; t < mj; ++t) {
        const double* p = density(jr[t]);
        for (std::size_t i = 0; i < nn; ++i)
            dj_in[i * mj + t] = p[i];
    }
    for (int t = 0; t < mk; ++t) {
        const double* p = density(kr[t]);
        for (std::size_t i = 0; i < nn; ++i)
            dk_in[i * mk + t] = p[i];
        scale[t] = requests[kr[t]].k_scale;
    }
    std::vector<std::vector<double>> js(nt), ks(nt);
    if (!cached()
        || !exact_jk_detail::try_contract_cached(
            pairs_, n_, rank_, ranks_, eri_, mj, mk, dj_in.data(), dk_in.data(),
            scale.data(), js, ks)) {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            js[tid].assign(nn * mj, 0.0);
            ks[tid].assign(nn * mk, 0.0);
            double* jt = js[tid].data();
            double* kt = ks[tid].data();
            const double* pj = dj_in.data();
            const double* pk = dk_in.data();
            const double* sk = scale.data();
            const int n = n_;
            const auto consume = [=](int a, int b, int c, int d, double v) {
                if (v == 0.0)
                    return;
                if (mj)
                    contract_j(n, mj, pj, jt, a, b, c, d, v);
                if (mk)
                    contract_k(n, mk, pk, kt, sk, a, b, c, d, v);
            };
            if (cached()) {
                // As in PySCF nr_incore, schedule canonical AO-pair rows in
                // small dynamic chunks, keeping heavy rows balanced across threads.
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
                for (std::int64_t ab = rank_;
                     ab < static_cast<std::int64_t>(n_) * (n_ + 1) / 2;
                     ab += ranks_) {
                    const int a =
                        static_cast<int>((std::sqrt(8.0 * ab + 1) - 1) / 2);
                    const int b =
                        ab - static_cast<std::int64_t>(a) * (a + 1) / 2;
                    const double* row = eri_.data() + ab * (ab + 1) / 2;
                    std::size_t cd = 0;
                    for (int c = 0; c <= a; ++c)
                        for (int d = 0; d <= (c == a ? b : c); ++d, ++cd)
                            consume(a, b, c, d, row[cd]);
                }
            } else {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (int left = rank_;
                     left < static_cast<int>(pairs_.pairs.size());
                     left += ranks_)
                    visit_quartets(pairs_, left, consume, bound, omega_);
            }
        }
    }
    const auto emit = [&](const std::vector<std::vector<double>>& parts, int m,
                          int t, RealMatrix* out) {
        RealMatrix M = xt::zeros<double>({n_, n_});
        for (int th = 0; th < nt; ++th)
            for (std::size_t i = 0; i < nn; ++i)
                M.data()[i] += parts[th][i * m + t];
#ifdef ATV_HAS_MPI
        if (ranks_ > 1) {
            // Only the calling thread enters MPI; OpenMP regions have
            // completed. Chunk counts to stay within the portable MPI
            // int-count interface.
            for (std::size_t offset = 0; offset < nn;) {
                const int count = static_cast<int>(std::min(
                    nn - offset,
                    static_cast<std::size_t>(std::numeric_limits<int>::max())));
                MPI_Allreduce(MPI_IN_PLACE, M.data() + offset, count,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                offset += count;
            }
        }
#endif
        if (c2s_.size())
            M = xt::linalg::dot(xt::transpose(c2s_),
                                RealMatrix(xt::linalg::dot(M, c2s_)));
        *out = std::move(M);
    };
    for (int t = 0; t < mj; ++t)
        emit(js, mj, t, requests[jr[t]].J);
    for (int t = 0; t < mk; ++t)
        emit(ks, mk, t, requests[kr[t]].K);
    ATV_DEBUG("[exact_jk] jk_ms={:.3f} densities={}", elapsed_ms(start), nr);
}

void ExactJK::compute(const RealMatrix& P, RealMatrix& J, RealMatrix& K) const {
    const JKRequest request{&P, &J, &K};
    compute(std::span<const JKRequest>(&request, 1));
}

void ExactJK::step(std::span<const JKRequest> requests) const {
    for (std::size_t r = 0; r < requests.size(); ++r) {
        if (requests[r].channel < 0)
            throw std::invalid_argument("ExactJK channel must be nonnegative");
        for (std::size_t q = 0; q < r; ++q)
            if (requests[q].channel == requests[r].channel)
                throw std::invalid_argument(
                    "ExactJK requests of one step need distinct channels");
    }
    if (cached()) {
        compute(requests);
        return;
    }
    // A channel's history is reusable only for the same J/K/scale shape.
    const auto usable = [&](const JKRequest& q) {
        if (static_cast<std::size_t>(q.channel) >= history_.size())
            return false;
        const History& h = history_[q.channel];
        return h.valid && (h.J.size() > 0) == (q.J != nullptr)
               && (h.K.size() > 0) == (q.K != nullptr)
               && (!q.K || h.k_scale == q.k_scale)
               && h.P.shape() == q.P->shape();
    };
    std::vector<RealMatrix> delta(requests.size());
    std::vector<JKRequest> work(requests.begin(), requests.end());
    for (std::size_t r = 0; r < requests.size(); ++r)
        if (usable(requests[r])) {
            delta[r] = *requests[r].P - history_[requests[r].channel].P;
            work[r].P = &delta[r];
        }
    compute(work);
    for (std::size_t r = 0; r < requests.size(); ++r) {
        const JKRequest& q = requests[r];
        if (static_cast<std::size_t>(q.channel) >= history_.size())
            history_.resize(q.channel + 1);
        History& h = history_[q.channel];
        if (usable(q)) {
            if (q.J)
                *q.J += h.J;
            if (q.K)
                *q.K += h.K;
        }
        h.P = *q.P;
        h.J = q.J ? *q.J : RealMatrix{};
        h.K = q.K ? *q.K : RealMatrix{};
        h.k_scale = q.k_scale;
        h.valid = true;
    }
}

void ExactJK::step(const RealMatrix& P, RealMatrix& J, RealMatrix& K,
                   int channel) const {
    const JKRequest request{&P, &J, &K, 1.0, channel};
    step(std::span<const JKRequest>(&request, 1));
}
} // namespace ATV::cpu::scf::internal
