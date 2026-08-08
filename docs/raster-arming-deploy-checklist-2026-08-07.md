# Raster Arming Overhaul — Deploy Checklist (2026-08-07)

Branches: GUI `fix/raster-reachability` @ `a107120` (105 camera-safe tests green) · BLACS `feat/raster-control-toggle` @ `d6f90d6` (worktree `.claude/worktrees/raster-control`).
Final whole-branch review: all cross-repo seams verified byte-for-byte; no path moves motors under Control=Local.

## Deploy order (spec §9 — do not swap)

1. **GUI first**: merge/checkout on the rastering `main` worktree, restart the rastering GUI.
2. **Then BLACS**: merge the RasteringDevice branch, restart BLACS (or the Rastering tab).
   Rationale: the new worker sends `disarm_raster` on connect under Control=Local; the new GUI answers it by releasing ownership. An old GUI would instead run `stop_raster()`.

## Operator verification (live rig — nothing here is testable headless)

Display (Task 3):
- [ ] Dead-zone shading visible over the camera image (z-order fix); recomputes on AOI change.
- [ ] Grey open circles (pending preview) vs blue filled dots (armed path) distinguishable at size 6.
- [ ] Direction lines vanish the moment a pattern is armed.
- [ ] After Stop, plot shows the cached preview (blank only if nothing was ever previewed); Stop is instant — no freeze on hull patterns.
- [ ] "Give to BLACS" third button face appears for a locally-held armed raster; BLACS's next `move_to_next` advances from the current point.

Worker (Task 5):
- [ ] BLACS restart under Control=Local: release reaches the GUI; an operator-armed path is NOT torn down.
- [ ] Drop counts appear in BLACS.log at both arm sites ("armed N / dropped M").
- [ ] `in_place`, `point_index=-1`, `frame`, `target_xy` land in the shot h5 without TypeError.
- [ ] Control=Local: tab restart AND queue abort log the coordinate-write skip and produce **no motor motion**; Control=BLACS: both still push coordinates.

Tab mirror (Task 6):
- [ ] Take local control at the GUI mid-queue → tab Control box unticks within ~1 s AND the next sequence with explicit coordinates raises.
- [ ] Continuous raster at the GUI + Raster Mode ticked → shots keep firing (SUCCESS acks, no queue pause).
- [ ] Continuous sweep running + "Give to BLACS" → sweep stops within one point, status line reads "Continuous run stopped - BLACS drives step-by-step from point N/M", and BLACS's next `move_to_next` advances from **that** point, not point 0.

## Open items (pending decisions)

1. ~~**Give-to-BLACS during a GUI continuous run currently halts the run.**~~ **RESOLVED** in `a9653c1` — operator ruled for the convert-deliberately option, and both halves shipped atomically: `give_remote_control()` clears `_raster_continuous` under the ownership lock with a status line naming the preserved cursor, and `arm_raster`'s already-armed branch refuses continuous→step with `raster_in_continuous_mode`. Spec §4 and the §6 error table updated.
2. GUI-side `request_move_*` handlers have no ownership gate (worker gates both paths today) — structural asymmetry, deserves its own design pass; not a merge blocker.
3. Near-duplicate arm tests (`test_zmq_v2_protocol.py:331` vs `:370`) — dedupe or pin `armed`/`dropped` in both.

## Notes

- `tests/CLAUDE.md` is an untracked auto-generated artifact — keep it out of merge commits.
- Never stage `calibration_data.json`.
- Full task-by-task record: `.superpowers/sdd/2026-08-07-raster-arming-remediation/progress.md` (git-ignored scratch; delete after merge).
