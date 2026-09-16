/// @file regressions/semiempirical/dftb2-multi-k-forces/reg_dftb2-multi-k-forces.cpp
/// @brief Analytic Bloch-sum nuclear forces + cell stress for multi-k
///        SCC-DFTB2, gated against re-SCC finite differences.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998) Eq. (15) carried to
/// k ≠ Γ; DFTB+ periodic force module (Hourahine et al., J. Chem. Phys. 152,
/// 124101 (2020)). Off-eq 1-D C-chain cell on a 2×1×1 Monkhorst–Pack mesh, so
/// the left/right lattice shells differ and F_x ≠ 0. In-tree mio-style SK
/// tables — no absolute DFTB+ identity is claimed. Public API only. <= 60 s.
/// Never shells to DFTB+.
#include <array>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// Off-eq 1-D C dimer: r=1.20 Å, Lx=2.8 Å → images at 1.20 and 1.60 Å (etc.).
cpu::Frame make_c_chain_offeq() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_type)[1] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.20;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 12.011;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Run DFTB2 on `g` at mesh `mesh`; energy [eV] out, forces [eV/Å] left on `g`,
/// stress [eV/Å³] Voigt left on `g.meta.get(kStress)`.
double run_dftb2(cpu::Frame& g, std::array<int, 3> mesh, bool with_forces,
                 bool with_stress = false) {
    g.meta.get(kEnergy) = 0;
    g.meta.get(kStress).fill(0.0);
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = with_forces;
    dftb.with_stress = with_stress;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.k_mesh = mesh;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(g);
    return g.meta.get(kEnergy);
}

double energy_at(double x0_shift, int atom, std::array<int, 3> mesh) {
    auto g = make_c_chain_offeq();
    g.span(kAtom_x)[atom] += static_cast<StorageReal>(x0_shift);
    return run_dftb2(g, mesh, /*with_forces=*/false);
}

/// Affine strain of cell + coordinates (mechanical ε), Voigt index `voigt`.
void apply_affine_strain(cpu::Frame& g, int voigt, double h) {
    Mat3 F = Mat3::identity();
    const int idx[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
    F(idx[voigt][0], idx[voigt][1]) += static_cast<AccReal>(h);
    F(idx[voigt][1], idx[voigt][0]) =
        F(idx[voigt][0], idx[voigt][1]); // symmetric ε
    Mat3 cell = g.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    g.box.set_cell(cell_new);
    for (int a = 0; a < g.nlocal(); ++a) {
        const AccReal rx = static_cast<AccReal>(g.span(kAtom_x)[a]);
        const AccReal ry = static_cast<AccReal>(g.span(kAtom_y)[a]);
        const AccReal rz = static_cast<AccReal>(g.span(kAtom_z)[a]);
        g.span(kAtom_x)[a] = static_cast<StorageReal>(
            F(0, 0) * rx + F(0, 1) * ry + F(0, 2) * rz);
        g.span(kAtom_y)[a] = static_cast<StorageReal>(
            F(1, 0) * rx + F(1, 1) * ry + F(1, 2) * rz);
        g.span(kAtom_z)[a] = static_cast<StorageReal>(
            F(2, 0) * rx + F(2, 1) * ry + F(2, 2) * rz);
    }
}

double energy_strained(int voigt, double h, std::array<int, 3> mesh) {
    auto g = make_c_chain_offeq();
    apply_affine_strain(g, voigt, h);
    return run_dftb2(g, mesh, /*with_forces=*/false);
}

} // namespace

int main() {
    constexpr std::array<int, 3> kMesh{{2, 1, 1}};
    constexpr std::array<int, 3> kGamma{{1, 1, 1}};

    auto frame = make_c_chain_offeq();
    const double e = run_dftb2(frame, kMesh, /*with_forces=*/true,
                               /*with_stress=*/true);
    std::printf("DFTB2 multi-k (2x1x1) C-chain forces: E=%.10f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    double f2 = 0;
    double net[3] = {0, 0, 0};
    double fx[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.10f, %.10f, %.10f) eV/A\n", i, fxi, fyi, fzi);
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
        fx[i] = fxi;
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        net[0] += fxi;
        net[1] += fyi;
        net[2] += fzi;
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(net[k]) > 1e-10) {
            std::printf("FAIL net force[%d]=%.3e > 1e-10 eV/A\n", k, net[k]);
            return 1;
        }
    }

    // Module boundary: analytic F = −∇E of the SAME lattice-summed, Bloch-
    // phased functional the multi-k SCC converged.
    constexpr double h = 1.0e-4;      // Å
    constexpr double kFdTol = 1.0e-5; // eV/Å (central-FD floor on multi-k SCC)
    for (int atom = 0; atom < 2; ++atom) {
        const double fd = -(energy_at(+h, atom, kMesh)
                            - energy_at(-h, atom, kMesh))
                          / (2.0 * h);
        std::printf("  F%dx analytic=%.10f  fd=%.10f  |d|=%.3e\n", atom,
                    fx[atom], fd, std::fabs(fx[atom] - fd));
        if (std::fabs(fx[atom] - fd) > kFdTol) {
            std::printf("FAIL F%dx vs FD: |d|=%.3e > %.1e eV/A\n", atom,
                        std::fabs(fx[atom] - fd), kFdTol);
            return 1;
        }
    }

    // Cell virial of the SAME image bond terms — the MP mesh is fractional, so
    // an affine strain enters only through the bonds. Self-images (A, A+T)
    // carry no force but do scale, and Lx = 2.8 Å < r_cut = 4.5 Å here, so
    // this gate is exactly what a molecule-in-a-box stress case cannot see.
    // kStress is the extensive Voigt virial [eV] (core/frame_schema.h);
    // these gates are written against the intensive stress W/V [eV/A^3].
    const auto& w_ext = frame.meta.get(kStress);
    double sigma[6];
    for (int v = 0; v < 6; ++v)
        sigma[v] = w_ext[static_cast<std::size_t>(v)] / frame.box.volume();
    constexpr double h_s = 2.0e-4;
    constexpr double kStressTol = 1.0e-6; // eV/Å³
    const double vol = frame.box.volume();
    if (!(vol > 0.0)) {
        std::printf("FAIL non-positive cell volume\n");
        return 1;
    }
    std::printf("  sigma=(%.10e, %.10e, %.10e, %.3e, %.3e, %.3e) eV/A^3\n",
                sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5]);
    for (int v = 0; v < 2; ++v) { // xx (chain axis), yy (vacuum axis)
        const double fd = -(energy_strained(v, +h_s, kMesh)
                            - energy_strained(v, -h_s, kMesh))
                          / (2.0 * h_s) / vol;
        std::printf("  sigma_%s analytic=%.10e  strain-fd=%.10e  |d|=%.3e\n",
                    v == 0 ? "xx" : "yy", sigma[v], fd,
                    std::fabs(sigma[v] - fd));
        if (!std::isfinite(sigma[v])
            || std::fabs(sigma[v] - fd) > kStressTol) {
            std::printf("FAIL sigma[%d] vs strain-FD: |d|=%.3e > %.1e\n", v,
                        std::fabs(sigma[v] - fd), kStressTol);
            return 1;
        }
    }
    for (int v = 2; v < 6; ++v) {
        if (!std::isfinite(sigma[v])) {
            std::printf("FAIL non-finite sigma[%d]\n", v);
            return 1;
        }
    }

    // The k-sampling must actually move the gradient, not just the energy.
    auto gamma_frame = make_c_chain_offeq();
    const double e_gamma = run_dftb2(gamma_frame, kGamma, /*with_forces=*/true);
    const double fx_gamma = static_cast<double>(gamma_frame.span(kAtom_fx)[0]);
    std::printf("  Γ (1x1x1): E=%.10f eV  F0x=%.10f eV/A  ΔF0x=%.6f\n", e_gamma,
                fx_gamma, fx[0] - fx_gamma);
    if (!std::isfinite(e_gamma) || std::fabs(fx[0] - fx_gamma) < 1e-3) {
        std::printf("FAIL multi-k force does not separate from Γ\n");
        return 1;
    }

    // Hard pins: ATV twin with full mio-1-1 C–C (re-pinned 2026-07-29).
    constexpr double kE_ref_eV = -91.8503968025;
    constexpr double kE_tol_eV = 0.05;
    constexpr double kF0x_ref = -11.1318394253;
    constexpr double kF_tol = 0.05;
    if (std::fabs(e - kE_ref_eV) > kE_tol_eV) {
        std::printf("FAIL energy pin: got %.10f want %.10f ± %.3f eV\n", e,
                    kE_ref_eV, kE_tol_eV);
        return 1;
    }
    if (std::fabs(fx[0] - kF0x_ref) > kF_tol) {
        std::printf("FAIL F0x pin: got %.10f want %.10f ± %.3f eV/A\n", fx[0],
                    kF0x_ref, kF_tol);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
