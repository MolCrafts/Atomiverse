# component-plugin-03-slot

oracle: harness

Expected:

```
ok: slot=adapter outer_iters=5 inner_converged=1
```

`on(phase).add_ref(inner)` occupies a `NestedLoopAdapter` Component slot. Outer loop runs 5 steps and does not converge; borrowed inner does.
