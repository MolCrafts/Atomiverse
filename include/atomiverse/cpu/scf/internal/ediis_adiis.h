/// @file cpu/scf/internal/ediis_adiis.h
/// @brief internal::EDIISMixer / internal::ADIISMixer — far-from-convergence
///        Fock mixers (energy-DIIS and augmented-Roothaan-Hall DIIS),
///        siblings of `internal::DIISMixer`.
///
/// **Internal SCF helpers.** Neither inherits `cpu::Component`; both are
/// owned by the mix stage inside the driver-ized RHF/UHF/RKS/UKS loop (and
/// by tests, via direct include of this header).
///
/// Both mixers store a short history of iterates and return the Fock
/// combination F̃ = Σ_i c_i F_i whose simplex-constrained coefficients
/// (Σ c_i = 1, c_i ≥ 0) minimize an energy-model objective — unlike Pulay's
/// CDIIS, whose unconstrained least-squares coefficients are only valid near
/// self-consistency. EDIIS/ADIIS are FAR-FIELD mixers: used to the end they
/// stall at ~1e-4 in the DIIS error, so production use escalates
/// EDIIS/ADIIS → CDIIS (see `MixStage`).
///
/// - **EDIIS** (K. N. Kudin, G. E. Scuseria, E. Cancès, J. Chem. Phys. 116,
///   8255 (2002), DOI 10.1063/1.1470195):
///     min_c  Σ_i c_i E_i − ½ Σ_ij c_i c_j ⟨D_i−D_j|F_i−F_j⟩
/// - **ADIIS** (X. Hu, W. Yang, J. Chem. Phys. 132, 054109 (2010),
///   DOI 10.1063/1.3304922), n = the latest iterate (energy-free):
///     min_c  2 Σ_i c_i ⟨D_i−D_n|F_n⟩ + Σ_ij c_i c_j ⟨D_i−D_n|F_j−F_n⟩
///
/// ⟨A|B⟩ = Tr(A·B), summed over both spin channels for unrestricted SCF.
/// The simplex constraint is handled by the t-parameterization
/// c_i = t_i²/Σ_j t_j² with a deterministic small-dimension minimizer
/// (projected-gradient descent with a fixed budget, dimension ≤ the history
/// depth) plus an exact KKT polish on the detected support. A degenerate
/// history (e.g. duplicated Fock matrices) returns the latest F — never NaN.
#pragma once

#include <deque>
#include <utility>
#include <vector>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief EDIIS: energy-DIIS Fock mixer over a (F, D, E) history.
///
/// DOI 10.1063/1.1470195. Dual-channel histories share one coefficient set
/// (spin-summed trace metric), mirroring `DIISMixer`.
class EDIISMixer {
  public:
    /// @param max_vectors Maximum number of stored (F, D, E) iterates.
    explicit EDIISMixer(int max_vectors = 6) : max_vectors_(max_vectors) {}

    /// @brief Store a new iterate and return the EDIIS-combined Fock pair.
    /// @param F_alpha α Fock matrix [Hartree].
    /// @param D_alpha α density matrix paired with `F_alpha` (dimensionless).
    /// @param F_beta  β Fock matrix [Hartree].
    /// @param D_beta  β density matrix paired with `F_beta` (dimensionless).
    /// @param energy  SCF total (or electronic — any constant shift drops
    ///                out under Σc = 1) energy of this iterate [Hartree].
    /// @return Combined (F̃_α, F̃_β) = Σ_i c_i (F_α,i, F_β,i) [Hartree].
    std::pair<RealMatrix, RealMatrix>
    step(const RealMatrix& F_alpha, const RealMatrix& D_alpha,
         const RealMatrix& F_beta, const RealMatrix& D_beta, double energy);

    /// @brief Single-channel convenience (restricted SCF; D is the total,
    ///        factor-2 density).
    /// @param F      Fock matrix [Hartree].
    /// @param D      Density matrix paired with `F` (dimensionless).
    /// @param energy SCF energy of this iterate [Hartree].
    /// @return Combined F̃ = Σ_i c_i F_i [Hartree].
    RealMatrix step(const RealMatrix& F, const RealMatrix& D, double energy);

    /// @brief Number of stored iterates.
    int size() const { return static_cast<int>(Fa_.size()); }

    /// @brief Clear all history.
    void reset();

    /// @brief Simplex coefficients of the most recent `step` (Σ = 1, ≥ 0).
    /// @return c, ordered oldest → newest; empty before any step.
    const std::vector<double>& coefficients() const { return c_; }

    /// @brief EDIIS objective at given coefficients over the CURRENT history
    ///        (test/diagnostic surface).
    /// @param c Coefficient vector, size == size(), Σc = 1 expected.
    /// @return Σ c_i E_i − ½ Σ_ij c_i c_j ⟨D_i−D_j|F_i−F_j⟩ [Hartree].
    double objective(const std::vector<double>& c) const;

  private:
    int max_vectors_;
    std::deque<RealMatrix> Fa_, Fb_, Da_, Db_;
    std::deque<double> E_;
    std::vector<double> c_;

    void push_(const RealMatrix& Fa, const RealMatrix& Da, const RealMatrix& Fb,
               const RealMatrix& Db, double energy);
    void solve_();
};

/// @brief ADIIS: augmented-Roothaan-Hall Fock mixer over a (F, D) history.
///
/// DOI 10.1063/1.3304922. Energy-free: the objective uses only density and
/// Fock differences against the latest iterate. Dual-channel histories share
/// one coefficient set (spin-summed trace metric), mirroring `DIISMixer`.
class ADIISMixer {
  public:
    /// @param max_vectors Maximum number of stored (F, D) iterates.
    explicit ADIISMixer(int max_vectors = 6) : max_vectors_(max_vectors) {}

    /// @brief Store a new iterate and return the ADIIS-combined Fock pair.
    /// @param F_alpha α Fock matrix [Hartree].
    /// @param D_alpha α density matrix paired with `F_alpha` (dimensionless).
    /// @param F_beta  β Fock matrix [Hartree].
    /// @param D_beta  β density matrix paired with `F_beta` (dimensionless).
    /// @return Combined (F̃_α, F̃_β) = Σ_i c_i (F_α,i, F_β,i) [Hartree].
    std::pair<RealMatrix, RealMatrix> step(const RealMatrix& F_alpha,
                                           const RealMatrix& D_alpha,
                                           const RealMatrix& F_beta,
                                           const RealMatrix& D_beta);

    /// @brief Single-channel convenience (restricted SCF; D is the total,
    ///        factor-2 density).
    /// @param F Fock matrix [Hartree].
    /// @param D Density matrix paired with `F` (dimensionless).
    /// @return Combined F̃ = Σ_i c_i F_i [Hartree].
    RealMatrix step(const RealMatrix& F, const RealMatrix& D);

    /// @brief Number of stored iterates.
    int size() const { return static_cast<int>(Fa_.size()); }

    /// @brief Clear all history.
    void reset();

    /// @brief Simplex coefficients of the most recent `step` (Σ = 1, ≥ 0).
    /// @return c, ordered oldest → newest; empty before any step.
    const std::vector<double>& coefficients() const { return c_; }

    /// @brief ADIIS objective at given coefficients over the CURRENT history
    ///        (test/diagnostic surface).
    /// @param c Coefficient vector, size == size(), Σc = 1 expected.
    /// @return 2Σ c_i⟨D_i−D_n|F_n⟩ + Σ c_i c_j⟨D_i−D_n|F_j−F_n⟩ [Hartree].
    double objective(const std::vector<double>& c) const;

  private:
    int max_vectors_;
    std::deque<RealMatrix> Fa_, Fb_, Da_, Db_;
    std::vector<double> c_;

    void push_(const RealMatrix& Fa, const RealMatrix& Da, const RealMatrix& Fb,
               const RealMatrix& Db);
    void solve_();
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
