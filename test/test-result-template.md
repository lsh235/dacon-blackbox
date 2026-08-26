# Test Result

## Run Metadata
- Date: 2026-08-27 Asia/Seoul
- Commit: repository has no commits yet
- Executor: Codex
- Environment: Ubuntu 24.04.4, Python 3.12.3, torch 2.8.0+cu128, RTX 4060

## Command Results
- Environment: `make -C development env-check` — PASS
- Build: `make -C development build` — PASS
- Spec smoke: `node test/run-spec-smoke.mjs` — PASS, 46 cases loaded
- Unit tests: `make -C development test` — PASS, 26 tests
- Runtime smoke: `make -C development smoke` — PASS
- Public data decode: 20 videos — PASS
- Existing PyTorch 2.4 checkpoints on PyTorch 2.8 — PASS, CSV unchanged
- New 1 epoch Stage 1/2/3 training and integrated GPU inference — PASS

## Case Outcomes
| ID | Result | Evidence |
|---|---|---|
| TC-001~TC-046 | PASS | Spec bundle loaded and mapped without structural errors |
| Environment contract | PASS | `development/tests/unit/test_environment.py` |
| Output schema | PASS | `development/tests/unit/test_contracts.py` |
| Submission ZIP | PASS | `development/tests/unit/test_submission.py` |
| Data and runtime guards | PASS | Data, inventory, checkpoint and manifest unit tests |

## Failures / Risks
- Official metrics and combined scoring remain unavailable.
- Full training data and external-data rules are unavailable.
- Stage 3 official 10Hz mapping remains unresolved.
- Accuracy, full-dataset runtime and L40S resource limits remain unverified.
