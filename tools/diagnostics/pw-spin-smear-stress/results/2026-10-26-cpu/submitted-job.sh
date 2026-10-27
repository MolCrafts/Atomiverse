#!/bin/bash -l
#SBATCH --job-name=dft-cpu-pw-record
#SBATCH --account=naiss2026-1-53-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=.claude/campaigns/dft-gap-wave2-20261026/cpu-pw/record-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load buildenv-gcc/2026.03-mpich
export RUSTUP_TOOLCHAIN=1.99.1-x86_64-unknown-linux-gnu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
pw_blas=/software/sse2/el9_epyc9005/manual/OpenBLAS/0.3.32/g14/hpc1/pthread/lp64/lib
export LD_LIBRARY_PATH="$pw_blas:$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
pw_out="$PWD/.claude/campaigns/dft-gap-wave2-20261026/cpu-pw"
pw_build="$pw_out/build"
.claude/scratch/hf-auto/env/bin/python "$pw_out/diagnostic-relink.py" > "$pw_out/diagnostic-relink-final.log" 2>&1
sha256sum "$pw_build/bin/pw-spin-smear-stress" "$pw_build/libatomiverse.a" > "$pw_out/final-before-sha256.txt"
for width in .02 .04 .12; do
 "$pw_build/bin/pw-spin-smear-stress" --mode probe --width "$width" --output "$pw_out/final-probe-$width.json" > "$pw_out/final-probe-$width.log" 2>&1
done
set +e
"$pw_build/bin/pw-spin-smear-stress" --mode matrix --width .02 --mp-width .04 --output "$pw_out/final-matrix.json" > "$pw_out/final-matrix.log" 2>&1
matrix_status=$?
set -e
"$pw_build/bin/pw-spin-smear-stress" --mode mpi --width .02 --output "$pw_out/final-mpi-one.json" > "$pw_out/final-mpi-one.log" 2>&1
srun --ntasks=2 --cpus-per-task=1 --cpu-bind=cores "$pw_build/bin/pw-spin-smear-stress" --mode mpi --width .02 --output "$pw_out/final-mpi-two.json" > "$pw_out/final-mpi-two.log" 2>&1
set +e
"$pw_build/bin/pw-spin-smear-stress-before" --mode negative --width .02 --output "$pw_out/final-before.json" > "$pw_out/final-before.log" 2>&1
before_status=$?
"$pw_build/bin/pw-spin-smear-stress" --mode cutoff --width .02 --output "$pw_out/final-cutoff-survey.json" > "$pw_out/final-cutoff-survey.log" 2>&1
cutoff_status=$?
set -e
if test "$before_status" -ne 1 || test "$cutoff_status" -ne 1; then
 echo "Expected original normalization and invalid recut reference checks to fail; got $before_status $cutoff_status"
 exit 1
fi
.claude/scratch/hf-auto/env/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('.claude/campaigns/dft-gap-wave2-20261026/cpu-pw')
a=json.loads((root/'final-mpi-one.json').read_text())['records'][0]
b=json.loads((root/'final-mpi-two.json').read_text())['records'][0]
energy=abs(a['energy_ha']-b['energy_ha'])
stress=max(abs(x-y) for x,y in zip(a['stress_ev_ang3'],b['stress_ev_ang3']))
assert energy < 1e-6,(energy,a,b)
assert stress < 2e-5,(stress,a,b)
before=json.loads((root/'final-before.json').read_text())
assert before['error']=='original stress violates physical simple-shear derivative',before
cutoff=json.loads((root/'final-cutoff-survey.json').read_text())
assert cutoff['error']=='hard-cutoff recut full-SCF strains are not a physical stress oracle',cutoff
(root/'final-mpi-comparison.json').write_text(json.dumps({'passed':True,'energy_error_ha':energy,'stress_error_ev_ang3':stress,'energy_gate_ha':1e-6,'stress_gate_ev_ang3':2e-5},indent=2)+'\n')
print('MPI serial/two-rank spin-smear stress passed',energy,stress)
PY
sha256sum "$pw_build/bin/pw-spin-smear-stress" "$pw_build/libatomiverse.a" > "$pw_out/final-after-sha256.txt"
cmp "$pw_out/final-before-sha256.txt" "$pw_out/final-after-sha256.txt"

if test "$matrix_status" -ne 0; then
 echo "Full matrix remains scientifically incomplete; all later evidence was collected. matrix_status=$matrix_status"
 exit "$matrix_status"
fi
