/// @file cpu/dynamics/virtual_sites.h
/// @brief General multi-topology virtual sites (TIP4P Weighted3 + TIP5P OutOfPlane3).
///
/// Massless sites are Frame atoms. Positions are reconstructed each step from
/// parents; forces on sites are redistributed (Feenstra / GROMACS-style) so
/// total force is preserved. Pair styles stay ordinary (`PairLJCut`,
/// `PairCoulCut`, …) — this is **not** a specialized tip4p pair.
///
/// Units: length Å, force eV/Å. No public Config bag; no factory free functions.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Table of virtual-site constructions + place / force-partition ops.
///
/// Register sites by stable tags, then either call `place` / `redistribute`
/// directly or bind the nested `Place` / `Redistribute` Components into a
/// Plan (MD defaults cover post_integrate / pre_force / post_force).
class VirtualSites {
  public:
    /// @brief Construction kind for one site.
    enum class Kind : std::uint8_t {
        Weighted3, ///< \(\mathbf{r}_s=(1-\alpha)\mathbf{r}_0+\frac\alpha2\mathbf{r}_1+\frac\alpha2\mathbf{r}_2\)
        OutOfPlane3, ///< \(\mathbf{r}_s=\mathbf{r}_0+a\mathbf{d}_1+b\mathbf{d}_2+c(\mathbf{d}_1\times\mathbf{d}_2)\)
    };

    VirtualSites() = default;

    /// @brief TIP4P-style M site: linear combination of O, H1, H2.
    /// @param site_tag Virtual site atom tag.
    /// @param p0 Parent 0 (oxygen) tag.
    /// @param p1 Parent 1 (H1) tag.
    /// @param p2 Parent 2 (H2) tag.
    /// @param alpha Dimensionless scale (\(\alpha=q_{\mathrm{dist}}/(\cos(\theta/2)\,r_{\mathrm{OH}})\)).
    void add_weighted3(TagInt site_tag, TagInt p0, TagInt p1, TagInt p2,
                       double alpha);

    /// @brief Out-of-plane site (TIP5P lone pair) from three parents.
    /// @param site_tag Virtual site atom tag.
    /// @param p0 Parent 0 (oxygen) tag.
    /// @param p1 Parent 1 (H1) tag.
    /// @param p2 Parent 2 (H2) tag.
    /// @param a Coefficient of \(\mathbf{d}_1=\mathbf{r}_1-\mathbf{r}_0\).
    /// @param b Coefficient of \(\mathbf{d}_2=\mathbf{r}_2-\mathbf{r}_0\).
    /// @param c Coefficient of \(\mathbf{d}_1\times\mathbf{d}_2\).
    void add_out_of_plane3(TagInt site_tag, TagInt p0, TagInt p1, TagInt p2,
                           double a, double b, double c);

    /// @brief \(\alpha = q_{\mathrm{dist}} / (\cos(\theta/2)\, r_{\mathrm{OH}})\).
    /// @param qdist O→M distance [Å].
    /// @param r_oh O–H bond length [Å].
    /// @param theta_hoh H–O–H angle [rad].
    /// @return Dimensionless α.
    /// @throws std::invalid_argument if geometry cannot form α.
    static double tip4p_alpha(double qdist, double r_oh, double theta_hoh);

    /// @brief Equilibrium TIP5P OutOfPlane3 coeffs for one lone-pair site.
    ///
    /// Solves \(\mathbf{r}_L-\mathbf{r}_O = a\mathbf{d}_1+b\mathbf{d}_2
    /// +c(\mathbf{d}_1\times\mathbf{d}_2)\) at rigid equilibrium geometry
    /// (Mahoney & Jorgensen TIP5P: \(r_{\mathrm{OL}}\), \(\theta_{\mathrm{LOL}}\)).
    ///
    /// @param r_ol O–L distance [Å] (TIP5P: 0.70).
    /// @param theta_lol L–O–L angle [rad] (TIP5P: 109.47°).
    /// @param r_oh O–H distance [Å] (TIP5P: 0.9572).
    /// @param theta_hoh H–O–H angle [rad] (TIP5P: 104.52°).
    /// @param sign +1 or −1 selects L1 vs L2 (sign of plane-normal component).
    /// @param[out] a,b,c Coefficients for `add_out_of_plane3`.
    /// @throws std::invalid_argument on non-physical inputs.
    static void tip5p_out_of_plane_coeffs(double r_ol, double theta_lol,
                                          double r_oh, double theta_hoh,
                                          int sign, double& a, double& b,
                                          double& c);

    /// @brief Reconstruct all registered site positions from parents [Å].
    void place(Frame& f);

    /// @brief Redistribute forces from each site onto its parents; zero site F.
    void redistribute(Frame& f);

    /// @brief Number of registered sites.
    int size() const { return static_cast<int>(sites_.size()); }

    /// @brief Component: place sites (MD: post_integrate + pre_force).
    class Place final : public Component {
      public:
        /// @param table Owning site table (must outlive this component).
        explicit Place(VirtualSites& table) : table_(&table) {}

        std::string name() const override { return "virtual_sites_place"; }

        std::vector<DefaultBinding> default_bindings() const override {
            return {ATV::make_binding(ATV::md::Phase::post_integrate, +100),
                    ATV::make_binding(ATV::md::Phase::pre_force, +50)};
        }

        void launch(const LaunchContext& ctx) override {
            table_->place(ctx.frame);
        }

      private:
        VirtualSites* table_ = nullptr;
    };

    /// @brief Component: force partition (MD: post_force).
    class Redistribute final : public Component {
      public:
        /// @param table Owning site table (must outlive this component).
        explicit Redistribute(VirtualSites& table) : table_(&table) {}

        std::string name() const override {
            return "virtual_sites_redistribute";
        }

        std::vector<DefaultBinding> default_bindings() const override {
            return {ATV::make_binding(ATV::md::Phase::post_force, +50)};
        }

        void launch(const LaunchContext& ctx) override {
            table_->redistribute(ctx.frame);
        }

      private:
        VirtualSites* table_ = nullptr;
    };

  private:
    struct Site {
        Kind kind = Kind::Weighted3;
        TagInt site = 0;
        TagInt p0 = 0, p1 = 0, p2 = 0;
        // Weighted3: alpha in param[0]
        // OutOfPlane3: a,b,c in param[0..2]
        double param[3] = {0.0, 0.0, 0.0};
        // Resolved local indices (rebuilt when n_total changes).
        int is = -1, i0 = -1, i1 = -1, i2 = -1;
    };

    void resolve_(Frame& f);

    std::vector<Site> sites_;
    int n_resolved_ = -1;
};

} // namespace md
} // namespace cpu
} // namespace ATV
