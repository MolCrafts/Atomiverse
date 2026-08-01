# Acceptance — gpw-bands-metallic

| Gate | Criterion |
| ---- | --------- |
| Product | Multi-k SCF with FD/MP smear + band path (frozen veff or documented product equivalent) |
| Character | Finite \(E_F\); metallic/path gates documented in REFERENCE |
| Oracle | CP2K 2026.1 SMEAR + BAND_STRUCTURE GTH-GPW twin in REFERENCE.md |
| Perf | Fair wall \(\le 10\times\) on matched twin |
| API | No public Config bags |

`acceptance_met=true` only when all gates pass.
