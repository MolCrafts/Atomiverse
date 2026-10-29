# Coherent CUDA final-state diagnostics

`pw-analytic-stress-cuda` retains the original fixed LSDA Si2 oracle. `pw-analytic-stress-cuda-family` runs the same private Driver, request, publication, failure/retry and frozen-state checks against four new actual QE7.6 references. All three entry points include `cuda_total_common.h`; no second Driver implementation or production diagnostic callback is introduced. Complete SCF diagnostics stay outside CTest.

After the five internal module files and this diagnostic package are integrated:

```sh
cmake --build build-gpu-job --target pw-analytic-stress-cuda pw-analytic-stress-cuda-family -j2
for mode in nested stress force energy nested-force nested-stress nested-energy exhausted retry; do
  build-gpu-job/bin/pw-analytic-stress-cuda --"$mode"
  for model in pbe pbesol revpbe rpbe; do
    build-gpu-job/bin/pw-analytic-stress-cuda-family "$model" --"$mode"
  done
done
```

The shared body uses immutable diagnostic Reference values, existing LSDA pins and `qe-gamma-pbe-fixed-2026-10-28/pins.h`. Four fixed-filling raw input/output/XML datasets, potential and binary hashes are beside the new pins. Geometry, GTH-PBE-q4 potential, 12Ry/48Ry, Gamma four occupied bands and mixing are unchanged. The independent QE oracle retains its original 1e-12Ry/250-step configuration; native method tol1e-10/max200 remain unchanged. Native cold constructor queries match base16x18x16 and GGA quadrature48x54x48. LSDA retains24x28x24. Original E/fullF/6sigma and frozen-C/ionic finite-difference gates remain in the common implementation.

The optional isolated builder never configures or writes the shared build:

```sh
python3 tools/diagnostics/pw-analytic-stress/build_cuda_total.py \
  --build build-gpu-job --archive build-gpu-job/libatomiverse.a \
  --output /path/to/lsda-output
python3 tools/diagnostics/pw-analytic-stress/build_cuda_total.py \
  --build build-gpu-job --archive build-gpu-job/libatomiverse.a \
  --family --output /path/to/family-output
```

It uses an existing `gpw-driver-uks-pbe` target's compiler/link dependencies; its manual device link is sm90. Ordinary CMake uses configured project architectures. `--include-root` can bind an independently staged internal header tree while testing an isolated archive. Each output includes actual source/headers/reference/archive/binary hashes and compile/link commands. Six GNU wrappers belong only to these diagnostics. `-Werror=return-type` applies only to their CXX compilation.

Force-only owner work avoids the GGA quadrature/gradient/W allocation. Energy-only performs no final sweep or owner allocation. Requested standalone F/W readbacks are public outputs. The external owner adapter additionally downloads C/rho/terms after `finish` solely for independent oracles; these arrays are not production transfers. External NVTX ranges distinguish owner finish, final bands and requested assembly. Existing Davidson scalar/eigen metadata,64B final residual/normalization and24B certification remain explicit; the method is not claimed zero-D2H. Cold radial/Ewald preparation remains on the host.

The first common-body extraction accidentally omitted ordinary-function return0, causing exit139; its original source and binaries remain a separate failed receipt. The corrected body explicitly returns success and was independently rebuilt and rerun for all45 modes. No old failed run is reclassified or inherited. These receipts qualify only Gamma RKS fixed LSDA/PBE-family scope, with no spin, smearing, multik or regulated endpoint extension.


The FD entry point is `pw-analytic-stress-cuda-fd MODEL MODE`, with the same
nine modes and model names lsda/pbe/pbesol/revpbe/rpbe. It uses the same shared
adapter, six bands and .02-Ry FD filling; five matched raw QE references are in
`qe-gamma-rks-fd-2026-10-28/`. The manual builder selects this entry with `--fd`.
The final spectrum is refilled at a shared chemical potential and every nonzero
occupation participates in final density. Auxiliary internal energy, entropy
correction and free energy are reported separately; publication retains the
reference SCF free energy and adds no entropy pressure.

The adapter snapshots every PWResult field before the owner runs, checks it
on normal and exceptional exits, and compares the retained method result.
It downloads final occupations and tests the Fermi law, Ne and entropy against
the final spectrum, and checks orbital density by direct C/f convolution.
Final residual/normalization metadata is two doubles per band (64 bytes for
fixed four-band cases, 96 bytes for six-band FD), plus the separate certificate.
The new pure density unit isolates tiny positive/signed tails from the dominant
density, preserves ordinary legacy density bitwise and rejects nonfinite final
occupations before mutation. It performs no SCF or XC evaluation.

Earlier fixed-only receipts and the failed exit139 diagnostic retain their
own source/binary scopes. Native current-library acceptance is required for
FD, broader spin/k settings and any later extension.
