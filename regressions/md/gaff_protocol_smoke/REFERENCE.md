# gaff_protocol_smoke

Short staged protocol on a charged 6-atom chain:

1. LBFGS minimize  
2. NVT Nose–Hoover 50 steps @ 400 K, dt=1 fs  
3. NPT MTK 50 steps with T ramp 400→500 K  

Success: all energies finite, box volume > 0. Wall budget: ≤120 s serial Release.

Not a paper-length equilibration.
