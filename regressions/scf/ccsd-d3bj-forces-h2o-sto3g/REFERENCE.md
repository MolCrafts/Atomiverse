# ccsd-d3bj-forces-h2o-sto3g

## What is gated

Product composition of public `RHF::ccsd_forces` with `d3bj = true`:

\[
\mathbf{F}(d3bj) - \mathbf{F}(no\ d3) = \mathbf{F}_{\mathrm{D3(BJ)}}
\]

and \(E(d3bj) - E(no\ d3) = E_{\mathrm{D3(BJ)}}\), on a **water dimer** so
the intermolecular D3 force is material.

Electronic CCSD gradients are already FD of the bare energy
(`ccsd-grad-h2o-sto3g`). This case only pins the dispersion leg of
`add_dispersion_forces` / `dispersion_energy` on the CCSD path.

## Geometry

Same water dimer as `mp2-d3bj-forces-h2o-sto3g` / `dftb2-dispersion-d3` [Å].

## Not claimed

Not a literature absolute of CCSD+D3; Grimme D3(BJ) model only.
