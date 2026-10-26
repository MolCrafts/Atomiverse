# LYP precision evidence

On 2026-10-25 the same native kernel and the unchanged XCvsLibxc battery
were compared with actually executed libxc 7.0.0 and 7.1.2 calls. This folder
contains diagnostic evidence, not a new oracle framework. `parity.cpp`
exercises all 20 original density/sigma pairs and 120 channels at the original
absolute/relative 1e-12 bounds; its logs also print raw external low-density
values at 17 digits.

The archived runs used the earlier native helper interface, which returned
LYP epsilon and converted it to f in the diagnostic caller. The current
`parity.cpp` consumes the canonical per-volume native output directly;
only the Libxc oracle multiplies its returned epsilon by rho. The historical
raw logs and their provenance hashes are unchanged. This caller migration
has not itself been rerun at this edit.

| sigma_aa=sigma_ab=sigma_bb | libxc 7.0.0 beta potential | libxc 7.1.2 beta potential | 90-digit symbolic derivative (rounded) |
| --- | ---: | ---: | ---: |
| 0.01 | 9.5415919508346363 | 9.5415919507723252 | 9.5415919508344597 |
| 0.1 | 95.464578041955633 | 95.464578041332558 | 95.464578041953984 |
| 1.0 | 954.69443895316556 | 954.69443894693609 | 954.69443895314923 |

See raw logs for all printed digits. The native direct-spin-fraction kernel
passes all 120 comparisons against 7.0.0 with maximum scaled error 1.77e-14.
Against 7.1.2 the same three beta-potential entries fail, maximum scaled
error 6.51e-12. No density point, cutoff, spin floor, functional parameter,
energy convention or test tolerance was changed.

`high_precision.py` differentiates the same Miehlich/LYP energy at 90 decimal
digits using mpmath 1.4.1. Its output agrees with the native/7.0.0 values;
it is an independent precision diagnosis, **not** another-program oracle
and not the source of test expectations. Reproduce with:

```sh
uv run --with mpmath==1.4.1 high_precision.py
```

The native algebra uses `4*rho_a*rho_b/rho^2`, `2*rho_a/rho`,
`2*rho_b/rho` and their direct derivatives instead of subtracting nearly
equal terms in `1-zeta^2`, `1+zeta` and `1-zeta`. The corresponding
reference expression is the official
[libxc 7.1.2 symbolic LYP source](https://gitlab.com/libxc/libxc/-/raw/7.1.2/python/gga_exc/gga_c_lyp.py).
The inspected
[generated 7.1.2 kernel](https://gitlab.com/libxc/libxc/-/raw/7.1.2/src/maple2c/gga_exc/gga_c_lyp.c)
has SHA256 `38239ef2f216310780dda61bd6c1265fd78d8739c3e52c248eff86f21846be68`.

In its `opz_pow_n_m1_k` helper, the value is `expm1(p*log1p(z))`, while
its derivative reconstructs the power by adding 1; downstream LYP terms
also add 1. For the minority spin fraction 0.019801980198019806, reconstructing
the 11/3 power this way differs from the direct power by 4.74e-11 relative.
`helper_cancellation.txt` records this double-precision diagnosis. This
explains why an upstream expression stabilized near z=0 can lose digits
near z=-1; it does not justify adjusting the native answer to the new error.

The direct external runs used the PySCF-bundled 7.0.0 library and native
Homebrew 7.1.2 library recorded, with hashes, in `provenance.json`. Compile
from the repository root with the matching include/library paths, for example:

```sh
clang++ -std=c++20 -O3 -march=native -Iinclude \
  -I/tmp/atv-xc-oracle/lib/python3.13/site-packages/pyscf/lib/deps/include \
  src/cpu/scf/xc_gga.cpp tools/diagnostics/lyp-precision/parity.cpp \
  /tmp/atv-xc-oracle/lib/python3.13/site-packages/pyscf/lib/deps/lib/libxc.15.dylib \
  -Wl,-rpath,/tmp/atv-xc-oracle/lib/python3.13/site-packages/pyscf/lib/deps/lib \
  -o /tmp/lyp-parity-7.0
/tmp/lyp-parity-7.0
```

For the 7.1.2 run the corresponding arguments were
`-I/opt/homebrew/include -L/opt/homebrew/lib -lxc`.
The main test build can select an isolated 7.0.0 CMake package explicitly;
no production dependency or runtime version branch is introduced.
