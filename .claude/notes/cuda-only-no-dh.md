<!-- mol:note:topic:cuda-only-no-dh -->
# CUDA-only iron law — no mid-path device↔host exchange

Binding for every claimed CUDA ES product path (RHF covered envelope, PW,
`cuda::gpw::GpwDft`, future methods).

## Law

Once a CUDA method enters its SCF (or the force/stress body of that step),
scientific state stays on the device.

| | Rule |
| --- | --- |
| **Forbidden** | Mid-iteration D2H or H2D of scientific fields (ρ, V_eff, V_AO, H, P, C, ε, forces, stress, per-iteration χ / weight packs); host loops reimplementing that science; silent CPU SCF fallback |
| **Allowed** | **Once** before the first iteration: geometry / basis / fixed H0 / tables H2D. **Once** after convergence or force finalization: Frame / `last_result` payload D2H |
| **Scalars** | Convergence metrics reduce to a device scalar; one small D2H of that scalar is fine if the host must decide stop (prefer mapped flags / device criteria) |

Violations are defects, not "hybrid footholds": staff a residency fix rather
than more dual-gates around the exchange. RHF s/p and PW density-side residency
are the acceptance bar.

## Classical MD (iron law 10)

The CUDA MD step loop follows the same law, stricter: **no device→host
synchronization at all** on the stepping thread between setup and the
end-of-run retrieve — not even a 4-byte flag. Rebuild, overflow, fault and stop
are decided on the device for the current step (conditional graph nodes,
capacity-sized grids reading device counts, fault-frozen state); the host
observes them late through pinned async copies and `cudaEventQuery`.
Thermo/recorder/analysis output is async and **lossless**: nothing an
asynchronous consumer is due is ever dropped.

**The one sanctioned exception — capture backpressure.** Every asynchronous
consumer of a driver (`cuda::AsyncConsumer`: `MrecRecorder`, `XyzRecorder`,
`Thermo`, `md::PressureSeries`, `DensityRecorder`, `RDF`, `MullikenObs`) reads
its steps from the driver's ONE shared capture buffer (`cuda::CaptureBuffer`,
owned by the Backend; the CPU twin is `cpu::CaptureBuffer`, both on the
host-side `ATV::CaptureLedger`). A step any consumer needs is frozen once, on
the compute stream, into a pinned host-mapped slot that every consumer due on
the step reads; the slot frees when the last of them released it. The byte
budget (`Driver::set_capture_buffer_bytes`, Python
`MD(capture_buffer_bytes=…)`, CMake `ATV_CAPTURE_BUFFER_BYTES`) bounds every
frame in flight across all consumers, and when it is full the run waits
until space frees. Precisely:
- *Where:* only on the stepping thread, in `Component::before_launch` (the
  host-side admission the backend calls for every in-loop launch that passes
  the cadence gate, before that step's work or step graph is enqueued — never
  inside a graph) or in the eager `setup_output` / `final_output`.
- *When:* only when the captures admitted but not yet released already fill
  the budget. With space, admission is host counters plus a mutex — no CUDA
  call, no device read, no wait (the cadence is the host step counter's, so
  the host knows when a capture is enqueued without asking the device; a
  consumer due on a step already captured joins it without a new frame).
- *How:* a host condition variable signalled by a consumer worker when it
  releases a frame — never `cudaStreamSynchronize` / `cudaEventSynchronize` /
  synchronous copies, never a busy-wait. The consumers keep draining while
  the stepping thread waits; a consumer failure wakes it and the run stops
  with the error (the backend also checks a host failure flag every step).
The capture kernels are graph-capturable (the label is the device step); each
consumer's worker polls its own completion word through mapped memory, off
the stepping thread. Evidence is an Nsight audit of steady steps
(`md-residency`, `benchmarks/perf/md/residency/`), not a source grep.
Open violations and fix order: [cuda-md-device-control.md](../specs/cuda-md-device-control.md).

## Open violations (fix order)

| Priority | Surface | Fix slug |
| --- | --- | --- |
| P1 | GPW UKS/MG/multi-k force payload and host assembly | `force-device-handoff-depth` |
| P1 | GPW multi-k payload / cold-start (host re-diag / densify) | `kpoint-payload-device` / `kpoint-device-cold-start` / `device-cold-start` |
| P1 | GPW multi-k setup residency (host χ / Löwdin / KB Vnl / H0) | `kpoint-device-{chi,orth,vnl}` / `h0-residency` |
| P2 | GPW stress free-fn host assembly | `device-stress` |

Γ RKS single-grid force handoff is closed within its declared envelope:
accepted P and raw-H W stay resident, AO/grid/ionic force contractions execute
on device, and immutable derivative tables upload before SCF. Actual Nsight
capture has zero H2D and only scalar eigensolver metadata D2H (4-byte status;
8-byte matrix max-abs confirmed by a separate CUPTI numerical probe).
The native run
included 45 units, five water cases, unchanged Driver/model gates and zero-error,
zero-leak memcheck. Cold setup, other force branches and stress remain open.

## Worker checklist (CUDA slug)

1. Load the CUDA kernel skills ([cuda-rhf-device-only.md](cuda-rhf-device-only.md)).
2. Grep the diff for `DeviceToHost`, `HostToDevice`, `download_*host`, mid-loop
   `cudaMemcpy` — zero scientific mid-path hits for green.
3. State the lever and which skill guided launch/tiling/memory.
4. A GH200 dual-gate is still required for correctness; it never excuses an
   exchange.

## Claims

- No "CUDA wins" on H₂/H₂O.
- No "device-only product" claim while a P0/P1 row is open on that path.
- Large-cell walls only after the method's mid-path exchange is gone.
