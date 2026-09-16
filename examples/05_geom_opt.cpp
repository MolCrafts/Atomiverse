/// @file examples/05_geom_opt.cpp
/// @brief Tutorial: Geometry optimization of aspirin with L-BFGS + RHF/STO-3G.
///
/// Demonstrates:
///   - Setting up a multi-atom molecule (aspirin, C9H8O4, 21 atoms)
///   - Configuring the SCF evaluator for ab initio forces
///   - Running L-BFGS geometry optimization with convergence monitoring
///   - Interpreting energy, forces, and displacement convergence criteria
///
/// Build:
///   cmake -B build -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
///   cmake --build build --target 05_geom_opt
///
/// Run:
///   ./build/examples/05_geom_opt
#include <cmath>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

/// @brief Build aspirin (C9H8O4) frame with approximate 3D coordinates.
///
/// Aspirin = acetylsalicylic acid. 21 atoms: 9 C + 8 H + 4 O.
/// Coordinates from PubChem CID 2244, slightly perturbed to give
/// the optimizer meaningful work.
///
/// STO-3G basis: 9*5 + 8*1 + 4*5 = 73 basis functions, 94 electrons.
static cpu::Frame make_aspirin() {
    // 21 atoms: C(6), H(1), O(8)
    cpu::Frame f(21);

    // Atomic numbers: 9C + 4O + 8H = 21 atoms, 94 electrons
    int types[21] = {
        6, 6, 6, 6, 6, 6, // C1-C6 (benzene ring)
        6,                // C7 (carboxyl C)
        8, 8,             // O8 (C=O), O9 (OH)
        8,                // O10 (ester O)
        6,                // C11 (acetyl carbonyl C)
        8,                // O12 (acetyl C=O)
        6,                // C13 (methyl C)
        1, 1, 1, 1,       // H14-H17 (ring)
        1,                // H18 (OH)
        1, 1, 1,          // H19-H21 (CH3)
    };

    // Masses [amu]
    double masses[21] = {
        12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011,
        15.999, 15.999, 15.999, 12.011, 15.999, 12.011, 1.008,
        1.008,  1.008,  1.008,  1.008,  1.008,  1.008,  1.008,
    };

    // Coordinates [Angstrom] — PySCF-validated aspirin geometry
    double coords[21][3] = {
        {1.2879, 0.1068, 0.0000},    // C1
        {0.1768, 1.0827, 0.0000},    // C2
        {0.5171, 2.4290, 0.0000},    // C3
        {1.8519, 2.7993, 0.0000},    // C4
        {2.9293, 1.8524, 0.0000},    // C5
        {2.6191, 0.4890, 0.0000},    // C6
        {-1.2076, 0.6698, 0.0000},   // C7
        {-1.5541, -0.4994, 0.0000},  // O8
        {-2.0632, 1.6805, 0.0000},   // O9
        {0.9973, -1.2285, 0.0000},   // O10
        {-0.2830, -1.7447, 0.0000},  // C11
        {-0.3696, -2.9537, 0.0000},  // O12
        {-1.3593, -0.7411, 0.3000},  // C13
        {-0.2980, 3.1365, 0.0000},   // H14
        {2.0791, 3.8586, 0.0000},    // H15
        {3.9560, 2.1939, 0.0000},    // H16
        {3.4546, -0.2019, 0.0000},   // H17
        {-2.9828, 1.3621, 0.0000},   // H18
        {-1.0697, -0.1755, 1.1920},  // H19
        {-2.3482, -1.1583, 0.5001},  // H20
        {-1.4267, -0.2091, -0.6502}, // H21
    };

    for (int i = 0; i < 21; ++i) {
        f.span(ATV::kAtom_type)[i] = types[i];
        f.span(ATV::kAtom_mass)[i] = double(masses[i]);
        f.span(ATV::kAtom_x)[i] = double(coords[i][0]);
        f.span(ATV::kAtom_y)[i] = double(coords[i][1]);
        f.span(ATV::kAtom_z)[i] = double(coords[i][2]);
    }
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

int main() {
    ATV::log_init(ATV::LogConfig{.level = spdlog::level::debug});
    ATV_INFO("=== Atomiverse: Aspirin Geometry Optimization ===");
    ATV_INFO("    Molecule: C9H8O4 (acetylsalicylic acid)");
    ATV_INFO("    Method:   RHF / STO-3G (rhf_config)");
    ATV_INFO("    Optimizer: L-BFGS (memory=5)");

    auto frame = make_aspirin();

    // Concrete RHF force provider — parameters are constructor arguments.
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                      /*tol=*/double(1e-6), /*max_iter=*/200};

    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{
        5,            // memory
        double(0.01), // alpha0 [Å²/eV] (only the Fixed-mode / step-0 fallback)
        20            // max_ls (line-search trial cap)
    });
    opt.on(ATV::minimizer::Phase::force).add(std::move(rhf));

    // Convergence is a POLICY, not an algorithm parameter: it lives on a
    // Criterion, not on the optimizer. `ForceConverged` requires ALL five
    // criteria simultaneously. These are looser than the built-in defaults,
    // which is the point of stating them — a tutorial that must finish in 25
    // steps cannot ask for a 0.002 Å displacement tolerance.
    //   ftol=0.05 eV/Å   (max force)   — typical quantum-chemistry threshold
    //   rms_ftol=0.03    (RMS force)
    //   max_dtol=0.05 Å  (max disp)
    //   rms_dtol=0.02 Å  (RMS disp)
    //   etol=1e-3 eV     (energy diff)
    opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.05, /*rms_ftol=*/0.03,
                                          /*max_dtol=*/0.05,
                                          /*rms_dtol=*/0.02, /*etol=*/1e-3});
    ATV_INFO("[example] optimizer build start");
    opt.build();
    ATV_INFO("[example] optimizer build done");

    // 25 steps keeps this tutorial short: each step is a full RHF/STO-3G SCF
    // + analytic gradient on all 21 atoms, so the cost is dominated by the
    // electronic-structure work, not the optimizer. The energy drops
    // steadily; raise the budget for tighter convergence.
    ATV_INFO("[example] optimization start max_steps=25");
    opt.run(frame, 25);

    // Report final state
    // Compute final max force
    double max_f = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        double fx = frame.span(ATV::kAtom_fx)[i],
               fy = frame.span(ATV::kAtom_fy)[i],
               fz = frame.span(ATV::kAtom_fz)[i];
        double mag = std::sqrt(fx * fx + fy * fy + fz * fz);
        if (mag > max_f)
            max_f = mag;
    }
    ATV_INFO("[example] final state energy_eV={:.6f} max_force_eVA={:.6f}",
             frame.meta.get(ATV::kEnergy), max_f);

    // Print final coordinates
    ATV_INFO("[example] final coordinates [Angstrom]");
    const char* symbols[] = {"", "H", "He", "Li", "Be", "B", "C", "N", "O"};
    for (int i = 0; i < frame.nlocal(); ++i) {
        int Z = static_cast<int>(frame.span(ATV::kAtom_type)[i]);
        ATV_INFO("  {:>2s} {:12.6f} {:12.6f} {:12.6f}", symbols[Z],
                 frame.span(ATV::kAtom_x)[i], frame.span(ATV::kAtom_y)[i],
                 frame.span(ATV::kAtom_z)[i]);
    }

    ATV_INFO("[example] done");
    return 0;
}
