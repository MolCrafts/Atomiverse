/// @file cpu/gpw/internal/gpw_workspace.h
/// @brief GpwWorkspace — solver-private scratch for one GPW SCF `run()`.
///
/// Owned by the GPW SCF runners in `gpw_solver.cpp` (Γ RKS/UKS and multi-k
/// RKS/UKS), never by `cpu::Frame` and never threaded through Build/Launch
/// contexts. Mirrors `cpu::pw::internal::PWWorkspace`: the workspace is the
/// solver's private state bag for geometry-fixed grids / product lists and
/// per-iteration effective potential. Step helpers and the forthcoming
/// `KsGridBuild` take it as an explicit parameter.
///
/// FFT plans stay in `fft_cache` (thread-local by grid shape), not here.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

/// @brief Private GPW SCF workspace (one instance per SCF path / `run()`).
///
/// Default-constructible. Callers fill geometry-fixed members once after
/// setup, then update `last_veff` each KS build. Empty `cbg` / `kbg` /
/// `products` / `mg.levels` means that path is unused for the current run
/// (e.g. multigrid leaves `cbg` empty; single-grid leaves `products` empty).
struct GpwWorkspace {
    /// Density / Poisson / XC fine grid (or multigrid finest level twin).
    UniformGrid grid{};

    /// CP2K-style multigrid levels (empty / single level when ngrids==1).
    MultiGrid mg{};

    /// Geometry-fixed primitive-pair product list (multigrid collocate path).
    ProductSet products{};

    /// Compact Γ basis-on-grid (single-grid path; empty under multigrid).
    CompactBasisGrid cbg{};

    /// Pre-summed χ^k on the fine grid (multi-k single-grid; empty under MG).
    KpointBasisGrid kbg{};

    /// Last fine-grid effective potential from the KS build [Hartree].
    /// RKS: length `grid.npts()`. UKS paths may pack α then β (length 2·npts).
    std::vector<double> last_veff{};
};

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
