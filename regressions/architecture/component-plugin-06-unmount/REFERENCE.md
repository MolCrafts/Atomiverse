# component-plugin-06-unmount

oracle: harness

```
ok: unfix-remount launches=6
```

`run(2)` → launches==3 (initial force evaluation + 2 steps) → `unfix("fake_eval")`
throws `missing required` + how-to → `add_ref` → `run(2)` → launches==6 (the
remount changes the force assembly, so forces are re-initialized before the
2 steps).
