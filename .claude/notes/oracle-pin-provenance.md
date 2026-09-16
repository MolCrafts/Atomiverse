<!-- mol:note:topic:oracle-pin-provenance -->
## Oracle pin provenance (binding)

**Why this exists.** Agents that want a green will invent pins: numpy rewrites
of ATV, self-golden `expected := what_we_got()`, literature windows labelled
`oracle: cp2k`, and **cross-representation** "twins" (e.g. GTH-GPW energies
used as if they validated NC-UPF plane-wave). Those never fail and never
measure the implementation.

### All oracle loops (pyscf / cp2k / metalwalls / tb)

1. **A pin is a number the other code printed.** Provenance = binary path +
   version/sha + input file + output line (or log excerpt). Re-runnable.
2. **`tools/*_oracle/` may only parse oracle output** — never reimplement the
   ATV algorithm, never seed from ATV's own measurement.
3. **`oracle:` names the code that produced the number**, not the paper you
   read. A formula from a paper is `oracle: literature` and only if the
   formula is from the paper, not from our header.
4. **No external number ⇒** `oracle: literature` with a real citation, or the
   slug is **not landable**. Never invent `oracle: <engine>`.
5. **Self-golden is forbidden.** Ask: if our implementation were wrong, would
   this number change? If no, it is not a pin.

**Historical failure (metalwalls):** four `oracle: metalwalls` pins were
numpy/ATV self-checks while MW was uninstalled; real MW later disagreed by
6–16%. Same pattern applies to CP2K "windows" and unrun `.inp` sketches.

### CP2K method loop — strict twin (supersedes soft practice)

**Rule:** for every landable `cp2k-method-loop` slug:

| Gate | Requirement |
| --- | --- |
| Install | **Install / use a real CP2K binary** on the pin host. "Module empty" /
  "not on PATH" is an action item (`tools/cp2k_oracle/` + local/source build),
  not a reason to skip. |
| Algorithm match | The offline twin uses **CP2K's algorithm for the same method
  class ATV is claiming** — same representation family (GPW↔GPW GTH/MOLOPT,
  etc.), same spin/k/XC class, comparable cutoff/basis/PP. ATV must be
  exercised on that matched path for the pin. |
| Correctness | Record numbers CP2K actually printed; regression asserts ATV vs
  those pins within stated tol. |
| Perf | **Must run** a fair wall on the same twin job:
  \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) written in `REFERENCE.md` (and
  `within_10x` / ratio gate when the loop requires it). |
| Literature escape | Only when the quantity is **not a CP2K algorithm** (e.g.
  a propagator CP2K does not implement) **or** a published number is the
  scientific standard: then `oracle: literature` + citation. **Not** a
  substitute for "we did not install CP2K". |

**Forbidden (no comparison value — do not land as CP2K pins):**

- GTH-GPW (or any Quickstep GTH) numbers used to "anchor" **NC-UPF / pure PW**
  ATV (or the reverse): unequal Hamiltonians are **not** twins.
- Any cross-PP, cross-basis-class, or cross-method "physics window" labelled
  `oracle: cp2k` while absolute or even relative digit gates pretend parity.
- `perf_exempt` because binary missing, inconvenient, or twin not run.
- `.inp` sketches never executed; "offline pin" without output lines.
- Self-FD of ATV total energy as the sole correctness gate while claiming
  `oracle: cp2k` (FD may be an extra consistency check only).

**If ATV's production path is representation-unequal to CP2K** for that
capability: either (a) land a **matched** ATV path that CP2K can twin (same
algorithm class), (b) pin to **literature**, or (c) leave the row **blocked /
not landable** — never a mismatched window.

**ctest** still never launches CP2K; pins are frozen offline. The obligation
is that those pins were produced by a real offline CP2K run (or real
literature), not that CI installs CP2K.

### Related

- CLAUDE.md § Oracle capability loops (thin router)
- `tools/cp2k_oracle/README.md`, `tools/metalwalls_oracle/`
- `docs/developer/oracle-capability-loop.md` (pattern; if it still says speed
  is optional or allows mismatch windows, **this note + CLAUDE win**)
