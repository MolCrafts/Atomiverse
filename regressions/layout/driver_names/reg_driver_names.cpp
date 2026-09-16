/// @file reg_driver_names.cpp
/// @brief backend-mirror-04-driver layout regression — the CPU loop node is
///        cpu::Driver<Phase> and the erased nestable base is cpu::DriverNode
///        (the backend-prefixed and old-base class names are gone). An
///        out-of-library consumer sees both under their new names and the
///        preset inheritance still holds.
#include "atomiverse/cpu/dynamics/md.h"

#include <cstdio>
#include <type_traits>

int main() {
    static_assert(
        std::is_base_of_v<ATV::cpu::Driver<ATV::md::Phase>, ATV::cpu::md::MD>,
        "MD must derive the renamed cpu::Driver<md::Phase> loop node");
    static_assert(std::is_base_of_v<ATV::cpu::DriverNode, ATV::cpu::md::MD>,
                  "MD must derive the erased base cpu::DriverNode");
    std::puts("driver_names: cpu::Driver<Phase> + cpu::DriverNode resolve");
    return 0;
}
