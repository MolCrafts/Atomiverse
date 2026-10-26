/// @file cpu/gpw/gpw_evaluator.h
/// @brief GPWEvaluator — GPW DFT energy component for CPU drivers.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/electronic/aspc_predictor.h"
#include "atomiverse/cpu/electronic/cpmd.h"
#include "atomiverse/cpu/electronic/xl_bomd.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/molpy.hpp"

namespace ATV {
namespace cpu {

namespace electronic {
class CarParrinello;
class XlBomdForce;
class SecondGenCp;
} // namespace electronic

namespace gpw {

/// @brief GPW energy / force / stress source for CPU drivers — the bind shell
///        around one `GpwDft`.
///
/// The method is the concrete `GpwDft` (kinds + every SCF parameter as its
/// members); this component only adapts it to a Driver slot: it re-reads the
/// Frame each launch, accumulates energy / forces / stress onto the Frame,
/// and hosts the AIMD propagators (XL-BOMD, CPMD, second-gen CP) that
/// `electronic::XlBomdForce` / `CarParrinello` / `SecondGenCp` switch on.
/// There is no second copy of the method parameters here.
///
/// ```cpp
/// cpu::gpw::GpwDft dft{kinds, /*cutoff_ry=*/280.0, cpu::gpw::XcKind::PBE};
/// md.on(md::Phase::force).add(cpu::gpw::GPWEvaluator(std::move(dft),
///                                                    /*compute_forces=*/true));
/// ```
///
/// **Variable cell:** `launch()` re-targets the method to the frame's current
/// orthorhombic box on every call (`GpwDft` rebuilds all geometry-dependent
/// structure — grids, lattice images, S/T/Vpp — per call), so cell-relaxation
/// drivers may rescale `frame.box` between launches. The MO warm start
/// survives a cell change (the AO basis is geometry-independent in size).
///
/// **Timestep:** under `cpmd = true` the electronic propagation shares the
/// nuclear clock, so the evaluator is a `cpu::TimestepConsumer` — `cpu::md::MD`
/// pushes the integrator's dt [fs] at build time. There is no duplicated
/// timestep member to keep in sync (and no way to run CPMD outside an MD loop).
class GPWEvaluator : public cpu::GPW, public cpu::TimestepConsumer {
  public:
    bool provides_force() const override { return compute_forces; }

    /// @brief Compute analytic forces (one SCF + one gradient pass), added to
    ///        `kAtom_f*` [eV/Å].
    bool compute_forces = false;
    /// @brief Compute cell stress (default: analytic fixed-\(P\) path).
    ///        Writes Frame `kStress` [eV/Å³] Voigt; also `last_stress_ha_bohr3()`.
    ///        With `xl_bomd_linearized`, each warm step uses the state-overload
    ///        shadow-density stress (no reference SCF); cold seed stays classic.
    bool compute_stress = false;
    /// @brief When true, use multi-SCF strain-FD (`gpw_stress_fd`) instead of
    ///        the product analytic path (`GpwDft::stress`). Default false.
    ///        Strain step / axis mask are the method's `stress_strain_h` /
    ///        `stress_axes`.
    ///        Refused with `xl_bomd_linearized` at `build()` (BO stress, not
    ///        the zero-SCF shadow functional).
    bool stress_use_fd = false;
    /// @brief Bind one GPW method to a Driver slot.
    /// @param method Configured `GpwDft` (kinds + parameters; not yet built —
    ///        the evaluator builds it against the Frame at `build()`).
    /// @param compute_forces Enable analytic forces.
    explicit GPWEvaluator(GpwDft method, bool compute_forces = false)
        : compute_forces(compute_forces), method_(std::move(method)) {}

    /// @brief The bound method (parameters are its members).
    GpwDft& method() { return method_; }
    /// @brief The bound method (read-only).
    const GpwDft& method() const { return method_; }

    std::string name() const override { return "gpw_evaluator"; }

    // ── Nuclear timestep, single-sourced from the MD integrator ──────────
    cpu::TimestepConsumer* as_timestep_consumer() override { return this; }
    /// @brief CPMD cannot run without the nuclear dt; BOMD/XL/SGCP do not
    ///        need it.
    /// @return True when `cpmd` is enabled.
    bool timestep_required() const override { return cpmd; }
    /// @brief Receive the nuclear MD timestep [fs] (pushed before `build()`).
    ///
    /// The electronic RATTLE/Verlet step of Car–Parrinello MD must run on the
    /// SAME clock as the nuclei, so the value is the integrator's own dt — not
    /// a second copy the user has to keep in sync.
    /// @param dt_fs Nuclear MD timestep [fs].
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    double last_energy_hartree() const { return last_.e_total; }
    /// @brief ⟨S²⟩ from the last UKS run (0 for closed-shell).
    double last_s2() const { return last_.s2; }
    const GpwResult& last_result() const { return last_; }
    /// @brief Last stress Voigt [Hartree/Bohr³] when `compute_stress` ran.
    const std::array<double, 6>& last_stress_ha_bohr3() const {
        return last_stress_ha_;
    }
    /// @brief Electronic kinetic energy \(T_e\) [eV] from the last CPMD
    ///        advance (0 when `cpmd` is off or before the first electronic step).
    double last_electronic_ke_ev() const { return last_te_ev_; }
    /// @brief Nosé–Hoover electronic thermostat energy \(E_{\mathrm{therm}}\)
    ///        [eV] from the last CPMD advance (0 when thermostat is off).
    double last_electronic_thermostat_ev() const { return last_etherm_ev_; }
    bool is_cpmd() const { return cpmd; }
    bool is_xl_bomd() const { return xl_bomd; }
    bool is_second_gen_cp() const { return second_gen_cp; }
    int xl_bomd_order() const { return xl_bomd_k; }

    // AIMD wrappers (types, not flags) configure private mode bits.
    friend class electronic::CarParrinello;
    friend class electronic::XlBomdForce;
    friend class electronic::SecondGenCp;

  private:
    // ── AIMD mode (private — use electronic::CarParrinello / XlBomdForce /
    //    SecondGenCp). Friends may write these; public API has no flags.
    bool xl_bomd = false;
    /// @brief XL-BOMD dissipation order K (Table I rows K = 3..9).
    int xl_bomd_k = 5;
    /// @brief Zero-SCF linearized-functional XL-BOMD forces (Niklasson
    ///        shadow Born–Oppenheimer potential). Requires `xl_bomd` and
    ///        `compute_forces`. After the cold full-SCF seed, each warm
    ///        force step builds \(H[n]\) from the auxiliary density \(n\),
    ///        diagonalizes once to \(D = q[n]\), writes the shadow energy
    ///        \(\mathcal{E}=\mathrm{Tr}(H D)-\mathrm{Tr}(H n)+E_{\mathrm{KS}}[n]\)
    ///        and assembles forces at \((D,H[n])\) with **no iterative
    ///        SCF** (`last_result().iters == 0`). Diagonalization paths:
    ///        Γ RKS (one total-P propagator, occ 2), Γ collinear UKS (dual
    ///        \(n^\alpha/n^\beta\), one diagonalization per spin, occ 1,
    ///        \(\mathcal{E}=\sum_\sigma[\mathrm{Tr}(H^\sigma D^\sigma)
    ///        -\mathrm{Tr}(H^\sigma n^\sigma)]+E_{\mathrm{KS}}[n]\)),
    ///        multi-k RKS (one complex \(n_k\) per sampled k), and multi-k
    ///        UKS (dual \(n^\sigma_k\)) with the k-weighted shadow energy
    ///        \(\mathcal{E}=\sum_k w_k\sum_\sigma\mathrm{Re}\,
    ///        \mathrm{Tr}[H^\sigma_k(D^\sigma_k-n^\sigma_k)]
    ///        +E_{\mathrm{KS}}[n]\) — `build()` throws for `use_ot`,
    ///        `cpmd`, or `second_gen_cp`. `compute_stress` is product:
    ///        each warm branch evaluates the analytic fixed-\(P\)
    ///        **shadow-density stress** (fixed-P stress at the warm payload
    ///        \(P = D = q[n]\), \(W\) from the \(H[n]\) eigenpairs) with
    ///        no reference SCF; `stress_use_fd` is refused (the multi-SCF
    ///        strain-FD oracle re-converges the SCF — Born–Oppenheimer
    ///        stress, not the shadow functional).
    bool xl_bomd_linearized = false;
    /// @brief Car–Parrinello MD: propagate occupied MO coefficients with a
    ///        fictitious mass and electronic RATTLE (not BOMD, not XL-BOMD).
    ///        After one cold SCF seed, each force step is a fixed-density KS
    ///        eval + electronic VV/RATTLE + analytic forces at that density.
    ///        Closed-shell RKS Γ, collinear Γ-UKS (dual α/β), multi-k RKS
    ///        (one `CpmdCplx` per sampled \(k\); Bloch \(C_k\)), or multi-k
    ///        collinear UKS (dual-spin Bloch \(C_k^\sigma\): one `CpmdCplx`
    ///        per \(k\) and spin, occ 1). `build()` throws with `use_ot`,
    ///        `xl_bomd`, `second_gen_cp`, or `compute_stress` (the CPMD
    ///        stress/variable-cell path is not yet productized). The
    ///        electronic thermostat is available on every branch, Γ and
    ///        multi-k alike (`cpmd_thermostat`).
    bool cpmd = false;
    /// @brief Fictitious electron mass \(\mu\) [a.u. electron mass]. Typical
    ///        CPMD values \(10^2\)–\(10^3\). Default 400.
    double cpmd_mu = 400.0;
    /// @brief Nosé–Hoover electronic thermostat on fictitious \(\dot C\)
    ///        (requires `cpmd = true`). Regulates \(T_e\) toward
    ///        `cpmd_ekin_target_ha` with coupling time `cpmd_tau_e`.
    ///        Γ: one NH per spin channel. Multi-k: one NH per (k, σ)
    ///        channel, each targeting the full \(K_0\); the published
    ///        totals are k-weighted (\(T_e = \sum_k w_k \sum_\sigma
    ///        T_e^{(k,\sigma)}\), same for \(E_{\mathrm{therm}}\)).
    ///        Default false (NVE electrons / pure CP).
    bool cpmd_thermostat = false;
    /// @brief Target fictitious electronic kinetic energy \(K_0\) [Hartree].
    ///        Must be > 0 when `cpmd_thermostat` is true.
    double cpmd_ekin_target_ha = 1e-5;
    /// @brief Electronic thermostat coupling time \(\tau_e\) [fs].
    double cpmd_tau_e = 50.0;
    /// @brief Second-generation Car–Parrinello MD (Kühne et al., PRL 98,
    ///        066401 (2007)): Kolafa ASPC density predictor + a **partial**
    ///        SCF corrector (not full BOMD, not fictitious-mass CPMD, not
    ///        Niklasson XL-BOMD).
    ///        - **Γ RKS:** one real total \(P\) (`ASPCPredictor`)
    ///        - **Γ UKS:** dual real \(P_\alpha, P_\beta\)
    ///        - **multi-k RKS:** one complex ASPC per sampled \(k\) on \(P_k\)
    ///        - **multi-k UKS:** dual complex ASPC per \(k\) (\(P_k^\alpha,
    ///          P_k^\beta\))
    ///        **OT corrector** is product at Γ when `use_ot=true` (RKS:
    ///        `second-gen-cpmd-ot`; UKS: `second-gen-cpmd-uks-ot`) and on
    ///        the multi-k paths (RKS: `second-gen-cpmd-ot-kpoints`; UKS:
    ///        `second-gen-cpmd-ot-uks-kpoints` — the per-k OT manifold(s)
    ///        warm-start from natural orbitals of the ASPC-predicted
    ///        \(P_k\) / dual-spin \(P_k^\sigma\)).
    ///        Mutually exclusive with `cpmd` / `xl_bomd`. Requires
    ///        `compute_forces=true`.
    bool second_gen_cp = false;
    /// @brief ASPC extrapolation order \(K\) (Kolafa 2004). Typical 1–4.
    int second_gen_cp_k = 3;
    /// @brief Corrector SCF iteration budget (≪ full BOMD `max_iter`).
    ///        Production steps after ASPC is warm use this cap.
    int second_gen_cp_max_iter = 5;

    /// Nuclear (and electronic) timestep [fs], pushed by the MD integrator
    /// through `set_timestep`. 0 = never pushed ⇒ `build()` refuses `cpmd`.
    double cpmd_dt_fs_ = 0.0;
    GpwResult last_{};
    GpwDft method_; ///< The one concrete method this shell binds.
    std::array<double, 6> last_stress_ha_{};
    /// XL-BOMD auxiliary-density propagator (engaged when `xl_bomd` on Γ).
    /// RKS: total \(P\). UKS: \(P_\alpha\). Empty when multi-k XL is active.
    std::optional<electronic::XlBomd> xl_;
    /// XL-BOMD β-spin propagator (Γ-UKS only; empty for RKS / multi-k).
    std::optional<electronic::XlBomd> xl_beta_;
    /// Multi-k XL α (or RKS total): one complex propagator per sampled k
    /// (empty on Γ).
    std::vector<electronic::XlBomdCplx> xl_k_;
    /// Multi-k UKS XL β: one complex propagator per k (empty for RKS / Γ).
    std::vector<electronic::XlBomdCplx> xl_k_beta_;
    /// Scratch for multi-k XL `guess_P_k` (lifetime covers `run`).
    std::vector<linalg::CplxMatrix> xl_guess_p_k_;
    /// Scratch for multi-k UKS XL `guess_P_k_beta` (lifetime covers `run`).
    std::vector<linalg::CplxMatrix> xl_guess_p_k_beta_;
    /// CPMD electronic propagator (engaged when `cpmd` on Γ). RKS: total occ.
    /// UKS: α channel (occ=1). Empty when multi-k CPMD is active.
    std::optional<electronic::Cpmd> cpmd_;
    /// CPMD β-spin propagator (Γ-UKS only; empty for RKS / multi-k).
    std::optional<electronic::Cpmd> cpmd_beta_;
    /// Multi-k CPMD α (or RKS total): one complex propagator per sampled k
    /// (empty on Γ).
    std::vector<electronic::CpmdCplx> cpmd_k_;
    /// Multi-k UKS CPMD β: one complex propagator per k (empty for RKS / Γ /
    /// the `n_beta == 0` half-metal edge).
    std::vector<electronic::CpmdCplx> cpmd_k_beta_;
    /// Scratch multi-k densities for fixed-density CPMD eval (RKS total, or
    /// UKS α with occ 1).
    std::vector<linalg::CplxMatrix> cpmd_guess_p_k_;
    /// Scratch multi-k UKS β densities for the dual-spin fixed-density eval.
    std::vector<linalg::CplxMatrix> cpmd_guess_p_k_beta_;
    double last_te_ev_ = 0.0;     ///< last electronic KE [eV]
    double last_etherm_ev_ = 0.0; ///< last electronic NH energy [eV]
    /// Second-gen CP ASPC density history (engaged when `second_gen_cp` on Γ).
    /// RKS: total \(P\). UKS: \(P_\alpha\). Empty when multi-k SGCP is active.
    std::optional<electronic::ASPCPredictor> sgcp_aspc_;
    /// Second-gen CP β-spin ASPC (Γ-UKS only; empty for RKS / multi-k).
    std::optional<electronic::ASPCPredictor> sgcp_aspc_beta_;
    /// Scratch for ASPC → column-major `guess_P` (lifetime covers `run`).
    linalg::RealMatrix sgcp_guess_p_;
    /// Scratch for ASPC β → column-major `guess_P_beta` (UKS).
    linalg::RealMatrix sgcp_guess_p_beta_;
    /// Multi-k SGCP α (or RKS total): one complex ASPC per sampled k (empty on Γ).
    std::vector<electronic::ASPCPredictorCplx> sgcp_aspc_k_;
    /// Multi-k SGCP β: one complex ASPC per k (empty for RKS / Γ).
    std::vector<electronic::ASPCPredictorCplx> sgcp_aspc_k_beta_;
    /// Scratch for multi-k SGCP `guess_P_k` (lifetime covers `run`).
    std::vector<linalg::CplxMatrix> sgcp_guess_p_k_;
    /// Scratch for multi-k UKS SGCP `guess_P_k_beta`.
    std::vector<linalg::CplxMatrix> sgcp_guess_p_k_beta_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
