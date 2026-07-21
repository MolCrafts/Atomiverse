/// @file cpu/scf/scf_solver.h
/// @brief **DELETED.** `SCFSolver` no longer exists. SCF is the
///        `CPUDriver<scf::Phase>` loop inside the concrete methods in
///        `methods.h` (`RHF`, `UHF`, `RKS`, `UKS`) — construct one, set
///        parameters as members, and call `run(frame)`.
#error                                                                         \
    "SCFSolver was deleted; SCF is the CPUDriver<scf::Phase> loop inside RHF/UHF/RKS/UKS — include atomiverse/cpu/scf/methods.h"
