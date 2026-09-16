/// @file cpu/scf/internal/cfmm_j.h
/// @brief internal::CfmmJ — continuous FMM Coulomb matrix J for Gaussian HF.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Shell-pair product distributions carry P_μν-weighted charges; extent
/// classification follows White–Johnson–Gill–Head-Gordon (Chem. Phys. Lett.
/// 230, 8 (1994), DOI 10.1016/0009-2614(94)01128-1). Far-field pairs use the
/// top-level `longrange/` multipole + M2L operators (link 20); near-field
/// pairs use the existing Schwarz-screened analytic ERI path. K never routes
/// through CFMM.
///
/// Includes only `atomiverse/cpu/longrange/*` (never `cpu/longrange/*`).
/// Class + constructor only — no `build_*` free function.
#pragma once

#include <utility>
#include <vector>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_operators.h"
#include "atomiverse/cpu/longrange/gaussian_extent.h"
#include "atomiverse/cpu/longrange/multipole_expansion.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Continuous-FMM workspace for the Coulomb matrix J[P] [Hartree].
///
/// Construct once per molecular shape; #prepare classifies shell-pair
/// distributions and builds complementary near/far interaction tables;
/// #j is called every SCF iteration with the current total density.
class CfmmJ {
  public:
    /// @brief Own copies of the Cartesian engine shells and build parameters.
    /// @param shells     Cartesian AO shells (internal::IntegralWorkspace::shells layout);
    ///                   centres in Bohr.
    /// @param order      Multipole truncation order p ≥ 0 (member
    ///                   `cfmm_order`; default 10). Controls M2L accuracy for
    ///                   the far-field walk.
    /// @param threshold  Extent / multipole threshold ε_thr ∈ (0, 1]
    ///                   (member `cfmm_threshold`, default 1e-9). Sets
    ///                   r_ext = √(2/ζ)·erfc⁻¹(ε) [Bohr] per WJGH 1994.
    /// @param schwarz_thr Cauchy–Schwarz pair threshold [Hartree^{1/2}],
    ///                   default 1e-10 (matches ShellPairList).
    /// @throws std::invalid_argument on bad order/threshold; std::runtime_error
    ///         if shells is empty.
    CfmmJ(const std::vector<native::Shell>& shells, int order, double threshold,
          double schwarz_thr = 1e-10);

    /// @brief Multipole truncation order p.
    int order() const { return order_; }

    /// @brief Extent threshold ε_thr (dimensionless).
    double threshold() const { return threshold_; }

    /// @brief Number of retained shell-pair product distributions.
    int n_distributions() const { return static_cast<int>(dists_.size()); }

    /// @brief One-time: build product distributions, extents, near/far tables.
    ///
    /// Near-field and far-field pair lists are complementary partitions of the
    /// canonical (d1 ≥ d2) distribution-pair space — no double counting.
    void prepare();

    /// @brief Build the Coulomb matrix J[P] = Σ_λσ (μν|λσ) P_λσ [Hartree].
    ///
    /// Starts from the exact Schwarz-screened direct J (ShellPairList::j),
    /// then for each far pair replaces the ERI contribution with the multipole
    /// M2L of P_μν-weighted s|s product charges. When #far_pairs is empty the
    /// result is exactly the Direct J.
    ///
    /// @param P_total Total density in the Cartesian engine AO space
    ///                [n_ao × n_ao, dimensionless] (RHF: includes occupation 2).
    /// @return Coulomb matrix [n_ao × n_ao, Hartree], symmetric.
    RealMatrix j(const RealMatrix& P_total) const;

    /// @brief Replace far-pair ERI contributions in an existing exact J with
    ///        multipole (M2L) contributions. No-op when #far_pairs is empty.
    ///
    /// Used by the RHF production path which builds the exact J via the fast
    /// shell-block route (G + ½K) then applies this correction.
    ///
    /// @param P_total Total density [n_ao × n_ao, dimensionless].
    /// @param J       Exact Coulomb matrix to correct in place [Hartree].
    void apply_far_correction(const RealMatrix& P_total, RealMatrix& J) const;

    /// @brief Unit monopole (overlap charge) of distribution d [dimensionless].
    ///
    /// For an s|s pair this equals ⟨χ_i|χ_j⟩ (the 0-th multipole moment about
    /// the product centre). Used by unit tests.
    /// @param d Distribution index in [0, n_distributions).
    double unit_monopole(int d) const;

    /// @brief Product-distribution centre of distribution d [Bohr].
    Vec3 distribution_center(int d) const;

    /// @brief True if distribution d is pure s|s (far-field eligible).
    bool distribution_spherical(int d) const;

    /// @brief Near-field (direct ERI) canonical distribution-pair list.
    const std::vector<std::pair<int, int>>& near_pairs() const {
        return near_pairs_;
    }

    /// @brief Far-field (multipole) canonical distribution-pair list.
    const std::vector<std::pair<int, int>>& far_pairs() const {
        return far_pairs_;
    }

    /// @brief AO indices (i ≥ j) of distribution d.
    std::pair<int, int> distribution_aos(int d) const;

  private:
    struct Dist {
        int i = 0;              ///< AO index i (i ≥ j)
        int j = 0;              ///< AO index j
        double schwarz = 0;     ///< √(ij|ij) [Hartree^{1/2}]
        double charge0 = 0;     ///< ∫ χ_i χ_j = overlap [dimensionless]
        Vec3 center;            ///< product centre [Bohr]
        double r_ext = 0;       ///< WJGH extent [Bohr]
        bool spherical = false; ///< both shells l == 0
    };

    void form_distributions_();
    void classify_pairs_();

    int n_ao_ = 0;
    int order_ = 10;
    double threshold_ = 1e-9;
    double schwarz_thr_ = 1e-10;
    std::vector<std::vector<double>> shell_data_; ///< owned [α..., c...]
    std::vector<native::Shell> shells_;
    std::vector<Dist> dists_;
    std::vector<std::pair<int, int>> near_pairs_;
    std::vector<std::pair<int, int>> far_pairs_;
    longrange::M2L m2l_;
    /// Schwarz pair list used for the exact near-field baseline (and the
    /// full-matrix path when every pair is near).
    ShellPairList sp_;
    bool prepared_ = false;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
