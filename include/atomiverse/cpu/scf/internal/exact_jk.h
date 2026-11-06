/// @file cpu/scf/internal/exact_jk.h
/// @brief Memory-budgeted exact Coulomb/exchange with eightfold ERI storage.
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV::cpu::scf::internal {

/// One density of a multi-density J/K build. A null J or K is not built, so
/// a request can be Coulomb-only, exchange-only or both; `k_scale`
/// multiplies the exchange during contraction (it is not applied to J).
struct JKRequest {
    const RealMatrix* P = nullptr; ///< Density, working AO basis.
    RealMatrix* J = nullptr;       ///< Out: J_ij = sum_kl (ij|kl) P_lk.
    RealMatrix* K = nullptr;       ///< Out: k_scale sum_jl (ij|kl) P_jl.
    double k_scale = 1.0;          ///< Exchange prefactor.
    int channel = 0;               ///< Incremental-history slot (step only).
};

/// One exact J/K engine: cache s8 integrals when they fit, otherwise evaluate
/// screened shell quartets during contraction. Both routes use the same
/// permutation contraction, including nonsymmetric GHF spin densities. Any
/// number of densities share one integral pass. The operator is 1/r, or
/// erf(omega r)/r for omega > 0 (range-separated exchange).
class ExactJK {
  public:
    /// Rebuild for a geometry. Budget is process memory in MB (95% admission);
    /// zero selects the automatic engine's bounded-memory execution.
    /// @param omega Range-separation parameter [1/Bohr]; 0 is bare Coulomb.
    /// @throws std::invalid_argument for a negative/nonfinite budget or
    ///         omega, or omega > 0 with only dense ERIs available.
    void prepare(const IntegralView& ints, double memory_mb = 4000.0,
                 bool mpi = false, double omega = 0.0);

    /// Contract every request in one integral pass. Inputs and outputs are in
    /// the working AO basis, including spherical AOs.
    void compute(std::span<const JKRequest> requests) const;
    /// Single-density J and K (both built, unit exchange scale).
    void compute(const RealMatrix& P, RealMatrix& J, RealMatrix& K) const;

    /// SCF update: cached ERIs contract full densities; the bounded-memory
    /// route contracts per-channel delta densities and accumulates, as in
    /// PySCF. A channel must keep the same J/K/scale shape between calls.
    void step(std::span<const JKRequest> requests) const;
    /// Single-density incremental J and K on one channel.
    void step(const RealMatrix& P, RealMatrix& J, RealMatrix& K,
              int channel = 0) const;

    std::size_t cache_bytes() const { return eri_.size() * sizeof(double); }
    bool cached() const { return !eri_.empty(); }
    bool prepared() const { return n_ > 0; }

  private:
    struct History {
        RealMatrix P, J, K;
        double k_scale = 1.0;
        bool valid = false;
    };

    int n_ = 0;
    double omega_ = 0.0;
    ShellBlockPairList pairs_;
    RealMatrix c2s_;
    std::vector<double> eri_;
    int rank_ = 0, ranks_ = 1;
    mutable std::vector<History> history_;
};

} // namespace ATV::cpu::scf::internal
