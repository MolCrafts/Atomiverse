/// @file reg_path_words.cpp
/// @brief backend-mirror-03-paths layout regression — backend subdirectories
///        are full English words. An out-of-library consumer reaches the MD and
///        minimize presets through the full-word include roots
///        `atomiverse/cpu/dynamics/` and `atomiverse/cpu/minimize/`; the
///        abbreviated legacy roots no longer exist.
///
/// Layout guarantee, not a physics value: it hard-codes "the full-word include
/// paths resolve and link" — compilation of these includes IS the contract.
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/minimize/minimize.h"

#include <cstdio>

int main() {
    std::puts("path_words: cpu/dynamics + cpu/minimize headers resolve on the "
              "full-word include root");
    return 0;
}
