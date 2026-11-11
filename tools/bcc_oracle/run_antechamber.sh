#!/bin/bash -l
# Re-run the AM1-BCC / ABCG2 oracle pins with AmberTools antechamber + sqm.
#
#   tools/bcc_oracle/run_antechamber.sh [outdir]      (default tools/bcc_oracle/outputs)
#
# Each case runs at the fixed input geometry (sqm `maxcyc=0`, no optimisation),
# AM1 closed shell, scfconv 1e-10. The captured per-case directory keeps the
# antechamber .ac output (final charges, column 9) and sqm.out (AM1 Mulliken
# base charges, "Mulliken Charge" table) — the two pinned quantities.
#
# The AmberTools Lmod module needs an easybuild toolchain stack that does not
# load on the login node, so the install prefix is used directly; override
# with AMBERHOME.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/outputs}"
module load buildenv-gcc/2026.03-mpich >/dev/null 2>&1
export AMBERHOME="${AMBERHOME:-/software/sse2/el9_epyc9005/easybuild/prefix/software/AmberTools/26.1-gmpich-2026.03}"
export PATH="$AMBERHOME/bin:$PATH"
export LD_LIBRARY_PATH="$AMBERHOME/lib:${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# name:net_charge
cases=("phenol:0" "acetate:-1")
for entry in "${cases[@]}"; do
    name="${entry%%:*}"
    nc="${entry##*:}"
    for method in bcc abcg2; do
        dir="$out/${name}_${method}"
        rm -rf "$dir" && mkdir -p "$dir"
        (
            cd "$dir"
            antechamber -i "$here/inputs/$name.mol2" -fi mol2 \
                -o "${name}_${method}.ac" -fo ac -c "$method" -nc "$nc" -pf y \
                -ek "qm_theory='AM1', maxcyc=0, scfconv=1.d-10" > antechamber.log 2>&1 \
                || { cat antechamber.log; exit 1; }
            rm -f sqm.pdb antechamber.log
        )
        echo "$name $method -> $dir/${name}_${method}.ac"
    done
done
