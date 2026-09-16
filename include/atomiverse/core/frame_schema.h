/// @file core/frame_schema.h
/// @brief Canonical semantic schema for Frame blocks and typed metadata.
///
/// The schema describes meaning and wire identity only.  It deliberately has
/// no CPU member-name column: every backend owns an independent Block/Column
/// representation and resolves these keys at its launch boundary.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "atomiverse/core/precision.h"

namespace ATV {

/// Runtime dtype of one Block column.  The tag is semantic; the wire schema
/// also fingerprints the configured sizeof(StorageReal/TagInt), so different
/// precision/ID builds cannot accidentally exchange payloads.
enum class ColDType : std::uint8_t { Storage, Acc, Tag, Int, Int8 };

constexpr std::size_t col_dtype_size(ColDType dtype) noexcept {
    switch (dtype) {
    case ColDType::Storage:
        return sizeof(StorageReal);
    case ColDType::Acc:
        return sizeof(AccReal);
    case ColDType::Tag:
        return sizeof(TagInt);
    case ColDType::Int:
        return sizeof(int);
    case ColDType::Int8:
        return sizeof(std::int8_t);
    }
    return 0;
}

/// Column-width semantics.  Runtime topology strides live in typed Frame
/// metadata and are materialized into concrete Column::cols() values.
enum class ColStride : std::uint8_t {
    One,
    Fixed3,
    BondPA,
    AnglePA2,
    AnglePA,
    DihedralPA3,
    DihedralPA,
    ImproperPA3,
    ImproperPA,
    MaxSpecial,
};

/// Counted section of an exchange row.
enum class ColGroup : std::uint8_t {
    Core,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Special
};

inline constexpr unsigned kColNone = 0;
inline constexpr unsigned kColCount = 1u << 0;
inline constexpr unsigned kColTopology = 1u << 1;
inline constexpr unsigned kColIdentityTail = 1u << 2;
/// @brief Declares the write-back set: the columns `to_molrs` returns to
///        the caller's Frame as native f64 (x/y/z, v, f, charge). Tag and
///        image stay a separate i32 identity path (narrowing + ix/iy/iz split).
inline constexpr unsigned kColWriteBack = 1u << 3;
inline constexpr unsigned kColMigrate = 1u << 4;
// Wire membership for the comm / border / reverse wires used to live here
// as kColComm / kColBorder / kColReverse — a global guess, made once, about
// what some consumer might want. It is now derived per run from what the
// mounted components declare (`Component::declare_ghosts`,
// `core/ghost_contract.h`).
// kColExchange stays a flag because migration carries what an atom IS, which
// is a property of the Frame rather than of whoever happens to be mounted.
inline constexpr unsigned kColExchange = 1u << 5;
inline constexpr unsigned kColState = kColMigrate | kColExchange;
/// @brief This column is a Cartesian coordinate and takes a PBC image
///        shift on the forward/border wires. x/y/z only. Never infer
///        this from `wire_id <= AtomFieldId::z` — a table insert ahead
///        of `z` would then silently add a box length to the new column.
inline constexpr unsigned kColPbcShift = 1u << 6;

/// One canonical field row. `wire_id` is the row index in
/// `ATV_ATOM_FIELDS` (and therefore of `AtomFieldId`). x/y/z are pinned
/// to 0/1/2 by static_assert below so a table reorder fails at compile
/// time rather than silently shifting the wrong column.
struct FieldSpec {
    std::uint16_t wire_id;
    const char* key;
    ColDType dtype;
    ColStride stride;
    ColGroup group;
    int epi;
    unsigned flags;
};

// id, key, element tier, dtype, stride, group, entries/item, flags
#define ATV_ATOM_FIELDS(X)                                                     \
    X(x, "x", StorageReal, Storage, One, Core, 0,                              \
      kColState | kColWriteBack | kColPbcShift)                                \
    X(y, "y", StorageReal, Storage, One, Core, 0,                              \
      kColState | kColWriteBack | kColPbcShift)                                \
    X(z, "z", StorageReal, Storage, One, Core, 0,                              \
      kColState | kColWriteBack | kColPbcShift)                                \
    X(vx, "vx", StorageReal, Storage, One, Core, 0, kColState | kColWriteBack) \
    X(vy, "vy", StorageReal, Storage, One, Core, 0, kColState | kColWriteBack) \
    X(vz, "vz", StorageReal, Storage, One, Core, 0, kColState | kColWriteBack) \
    X(fx, "fx", AccReal, Acc, One, Core, 0, kColState | kColWriteBack)         \
    X(fy, "fy", AccReal, Acc, One, Core, 0, kColState | kColWriteBack)         \
    X(fz, "fz", AccReal, Acc, One, Core, 0, kColState | kColWriteBack)         \
    X(mass, "mass", StorageReal, Storage, One, Core, 0, kColState)             \
    X(charge, "charge", StorageReal, Storage, One, Core, 0,                    \
      kColState | kColWriteBack)                                               \
    X(type, "type", int, Int, One, Core, 0, kColState)                         \
    X(tag, "tag", TagInt, Tag, One, Core, 0, kColState | kColIdentityTail)     \
    X(image, "image", std::int8_t, Int8, Fixed3, Core, 0, kColState)           \
    X(num_bond, "num_bond", int, Int, One, Bond, 0, kColState | kColCount)     \
    X(bond_atom, "bond_atom", TagInt, Tag, BondPA, Bond, 1,                    \
      kColState | kColTopology)                                                \
    X(bond_type, "bond_type", int, Int, BondPA, Bond, 1,                       \
      kColState | kColTopology)                                                \
    X(num_angle, "num_angle", int, Int, One, Angle, 0, kColState | kColCount)  \
    X(angle_atom, "angle_atom", TagInt, Tag, AnglePA2, Angle, 2,               \
      kColState | kColTopology)                                                \
    X(angle_type, "angle_type", int, Int, AnglePA, Angle, 1,                   \
      kColState | kColTopology)                                                \
    X(num_dihedral, "num_dihedral", int, Int, One, Dihedral, 0,                \
      kColState | kColCount)                                                   \
    X(dihedral_atom, "dihedral_atom", TagInt, Tag, DihedralPA3, Dihedral, 3,   \
      kColState | kColTopology)                                                \
    X(dihedral_type, "dihedral_type", int, Int, DihedralPA, Dihedral, 1,       \
      kColState | kColTopology)                                                \
    X(nspecial, "nspecial", int, Int, Fixed3, Special, 0,                      \
      kColState | kColCount)                                                   \
    X(special, "special", TagInt, Tag, MaxSpecial, Special, 1,                 \
      kColState | kColTopology)                                                \
    X(num_improper, "num_improper", int, Int, One, Improper, 0,                \
      kColState | kColCount)                                                   \
    X(improper_atom, "improper_atom", TagInt, Tag, ImproperPA3, Improper, 3,   \
      kColState | kColTopology)                                                \
    X(improper_type, "improper_type", int, Int, ImproperPA, Improper, 1,       \
      kColState | kColTopology)

enum class AtomFieldId : std::uint16_t {
#define ATV_FIELD_ENUM(id, key, Elem, DTag, Stride, Group, epi, flags) id,
    ATV_ATOM_FIELDS(ATV_FIELD_ENUM)
#undef ATV_FIELD_ENUM
        COUNT
};

inline constexpr FieldSpec kAtomFieldSpecs[] = {
#define ATV_FIELD_SPEC(id, key, Elem, DTag, Stride, Group, epi, flags)         \
    FieldSpec{static_cast<std::uint16_t>(AtomFieldId::id),                     \
              key,                                                             \
              ColDType::DTag,                                                  \
              ColStride::Stride,                                               \
              ColGroup::Group,                                                 \
              epi,                                                             \
              flags},
    ATV_ATOM_FIELDS(ATV_FIELD_SPEC)
#undef ATV_FIELD_SPEC
};

inline constexpr int kAtomFieldCount =
    static_cast<int>(sizeof(kAtomFieldSpecs) / sizeof(FieldSpec));
inline constexpr std::uint16_t kDynamicFieldId = 0xffffu;

static_assert(static_cast<std::uint16_t>(AtomFieldId::x) == 0
                  && static_cast<std::uint16_t>(AtomFieldId::y) == 1
                  && static_cast<std::uint16_t>(AtomFieldId::z) == 2,
              "x/y/z must stay the first three AtomFieldId values so a "
              "PBC-shifted column's wire_id is the axis index; reorder the "
              "table and this assert fails instead of silently shifting a "
              "new column");

/// Compile-time typed column key.  Dynamic columns use kDynamicFieldId and
/// are still dtype-checked by Block::span.
template <typename T>
struct ColKey {
    const char* block;
    const char* key;
    std::uint16_t wire_id;
    ColDType dtype;
};

#define ATV_FIELD_KEY(id, key, Elem, DTag, Stride, Group, epi, flags)          \
    inline constexpr ColKey<Elem> kAtom_##id{                                  \
        "atoms", key, static_cast<std::uint16_t>(AtomFieldId::id),             \
        ColDType::DTag};
ATV_ATOM_FIELDS(ATV_FIELD_KEY)
#undef ATV_FIELD_KEY

/// A typed metadata key.  `serializable` is part of the semantic schema;
/// backend-only entries (population/capacity/index caches) never cross FFI.
template <typename T>
struct MetaKey {
    const char* key;
    bool serializable;
};

using MetaBool3 = std::array<bool, 3>;
using MetaI32x3 = std::array<std::int32_t, 3>;
using MetaI64x3 = std::array<std::int64_t, 3>;
using MetaU32x3 = std::array<std::uint32_t, 3>;
using MetaU64x3 = std::array<std::uint64_t, 3>;
using MetaF32x3 = std::array<float, 3>;
using MetaF64x3 = std::array<double, 3>;
using MetaF32x6 = std::array<float, 6>;
using MetaF64x6 = std::array<double, 6>;
using MetaF32x9 = std::array<float, 9>;
using MetaF64x9 = std::array<double, 9>;
using MetaReal6 = MetaF64x6;

inline constexpr MetaKey<std::int64_t> kNLocal{"nlocal", false};
inline constexpr MetaKey<std::int64_t> kNGhost{"nghost", false};
inline constexpr MetaKey<std::int64_t> kNMax{"nmax", false};
inline constexpr MetaKey<std::int64_t> kBondPerAtom{"bond_per_atom", false};
inline constexpr MetaKey<std::int64_t> kAnglePerAtom{"angle_per_atom", false};
inline constexpr MetaKey<std::int64_t> kDihedralPerAtom{"dihedral_per_atom",
                                                        false};
inline constexpr MetaKey<std::int64_t> kImproperPerAtom{"improper_per_atom",
                                                        false};
inline constexpr MetaKey<std::int64_t> kMaxSpecial{"maxspecial", false};
inline constexpr MetaKey<std::int64_t> kHasBonded{"has_bonded", false};
inline constexpr MetaKey<std::int64_t> kNextTag{"next_tag", false};

inline constexpr MetaKey<std::int64_t> kStep{"step", true};
inline constexpr MetaKey<double> kEnergy{"energy", true};
/// Molecular dipole moment [Debye], written by cpu::MultipoleObserver.
inline constexpr MetaKey<MetaF64x3> kDipole{"dipole", true};
/// Traceless Buckingham quadrupole [D*Angstrom], packed xx,xy,xz,yy,yz,zz.
inline constexpr MetaKey<MetaF64x6> kQuadrupole{"quadrupole", true};
/// Multipole expansion origin (center of nuclear charge) [Angstrom].
inline constexpr MetaKey<MetaF64x3> kMultipoleOrigin{"multipole_origin", true};
inline constexpr MetaKey<double> kEnergyPair{"energy_pair", true};
inline constexpr MetaKey<double> kEnergyBond{"energy_bond", true};
inline constexpr MetaKey<double> kEnergyAngle{"energy_angle", true};
inline constexpr MetaKey<double> kEnergyDihedral{"energy_dihedral", true};
inline constexpr MetaKey<double> kEnergyImproper{"energy_improper", true};
/// Charging-work bookkeeping \(-\sum_i \Psi_i Q_i\) [eV] (MetalWalls JOSS eq. 6).
inline constexpr MetaKey<double> kEnergyChargingWork{"energy_charging_work",
                                                     true};
/// @brief Electronic-structure share of `kEnergy` [eV] — an **accumulator**
///        channel, written `+=` by every in-plan electronic-structure source
///        (Gaussian SCF, GPW, plane-wave, AM1/RM1, DFTB2) and cleared by
///        `ZeroEnergy` in lockstep with `kEnergy`.
///
/// Deliberately ONE key shared by every electronic method: co-bound electronic
/// sources legitimately merge into a single "electronic share" (Δ-learning, the
/// QM region of a QM/MM partition), so per-method keys would be key explosion
/// with no consumer. With a sole electronic source it equals `kEnergy`;
/// co-bound with classical terms it is the attributable QM part.
inline constexpr MetaKey<double> kEnergyElectronic{"energy_electronic", true};
inline constexpr MetaKey<double> kKinetic{"kinetic", true};
inline constexpr MetaKey<double> kConserved{"conserved", true};
/// Fictitious CP electronic kinetic energy \(T_e\) [eV] (CPMD evaluators;
/// adiabaticity monitor).
inline constexpr MetaKey<double> kEnergyElectronKinetic{
    "energy_electron_kinetic", true};
/// Nosé–Hoover electronic thermostat energy [eV] (0 for NVE electrons).
inline constexpr MetaKey<double> kEnergyElectronThermostat{
    "energy_electron_thermostat", true};
/// @brief Configurational stress: the **extensive Voigt virial**
///        \(W_{ab}=\sum_{i<j} r_{ij,a}F_{ij,b}\), Voigt order
///        (xx, yy, zz, xy, xz, yz), in **eV**, **compression-positive**,
///        **excluding** the kinetic term (that is `kStressKinetic`).
///
/// Extensive is forced, not chosen. Every evaluator `+=`-accumulates onto this
/// one key, and the driver MPI-`Reduce`s it by summing rank-local partials —
/// both are defined only for an extensive quantity (summing rank-local
/// *intensive* stresses would multiply σ by the rank count).
///
/// Consequences for producers and consumers:
/// - Classical pair/bond/long-range writers already accumulate \(W\) directly.
/// - Methods that naturally compute the intensive
///   \(\sigma_{ab}=-(1/\Omega)\,\partial E/\partial\varepsilon_{ab}\)
///   (PW, GPW, DFTB2) multiply by the cell volume **at the Frame write**; their
///   own `last_stress()` / `pw_stress()` returns stay intensive [eV/Å³].
/// - Every consumer divides by \(V\) to obtain a pressure: `IsotropicCellRelax`
///   and `JointIsotropicRelax` use \(\mathrm{tr}(W)/(3V)\),
///   `AnisotropicCellRelax` and `TriclinicCellRelax` use \(W_{ab}/V\), and
///   `StressConverged` judges \(W_{dd}/V\) against a target in eV/Å³.
///
/// Sign and factor follow Nielsen & Martin, Phys. Rev. B **32**, 3780 (1985),
/// Eqs. (5)–(6) (\(\sigma_{\alpha\beta}=T_{\alpha\beta}/\Omega\)); the DFT
/// strain-derivative form and the classical virial are the *same tensor* up to
/// \(1/\Omega\), since \(\partial E/\partial\varepsilon_{ab}=-W_{ab}\).
inline constexpr MetaKey<MetaF64x6> kStress{"stress", true};
/// @brief Kinetic stress \(\sum_i m_i v_{ia} v_{ib}\) [eV] — extensive, same
///        Voigt order and sign convention as `kStress`, kept on a separate key
///        so a 0 K geometry optimization can ignore it.
inline constexpr MetaKey<MetaF64x6> kStressKinetic{"stress_kinetic", true};
/// Per-electrode total charge [e], first ≤3 electrodes (ElectrodeChargeObserver).
inline constexpr MetaKey<MetaF64x3> kElectrodeCharge{"electrode_charge", true};
/// Differential capacitance \(C_{\mathrm{diff}}=\mathrm{Var}(Q)/(k_B T)\) [e/V]
/// from electrode charge fluctuations (CapacitanceChargeFluctuation).
inline constexpr MetaKey<double> kCapacitanceDiff{"capacitance_diff", true};
/// Integral capacitance \(C_{\mathrm{int}}=\langle Q\rangle/\Delta\Psi\) [e/V]
/// when \(\Delta\Psi\) is supplied (CapacitanceChargeFluctuation).
inline constexpr MetaKey<double> kCapacitanceIntegral{"capacitance_integral",
                                                      true};
/// Planar slab potential drop \(\Psi(z_{\max})-\Psi(z_{\min})\) [V]
/// (PotentialProfile observer; left-edge gauge \(\Psi(z_{\min})=0\)).
inline constexpr MetaKey<double> kPotentialDrop{"potential_drop", true};

} // namespace ATV
