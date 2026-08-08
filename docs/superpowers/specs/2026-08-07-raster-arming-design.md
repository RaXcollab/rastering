# Raster arming overhaul — design

**Date:** 2026-08-07
**Repos touched:** `RaXcollab/rastering` (GUI) and `labscript-suite` parent (`userlib/user_devices/RasteringDevice/`)
**Origin:** the 2026-08-07 "motor out of bounds" incident (root cause in §1)

---

## 1. Why

A raster sequence failed every shot with `[raster_step_failed] Rejected: motor out of
bounds`, for points the operator could see were nowhere near the travel limits, and which
manual moves reached fine. Restarting the BLACS tab changed nothing. Clicking **Auto
Raster** at the GUI cleared it instantly, and remote stepping then worked.

Root cause: **the armed path was not the path on screen, and nothing said so.**

Four mechanisms compose into that:

1. `start_raster` materializes the path once into `_raster_path_pts`
   (`raster_controller.py:1409-1419`). Nothing re-derives it.
2. `arm_raster` on an already-armed raster flips flags and returns SUCCESS without
   rebuilding (`raster_controller.py:478-503`). The BLACS tab calls `arm_raster` on every
   reconnect, so a tab restart re-armed the same stale path and reported success.
3. `_on_raster_param_changed` returns early while armed (`ui.py:1547-1548`), freezing the
   path preview — but hull vertex dots keep updating on every click (`ui.py:328-333`). The
   operator draws a new hull, sees new dots, and reads coordinates that are not being
   commanded.
4. `raster_step` pops the point under the lock and *then* enqueues the move
   (`:1478`, `:1518`). A rejected move does not rewind the cursor, so every retry hits a
   different unreachable point and produces an identical error — no progress, ever.

Why those points were unreachable: with the 2026-07-20 fit on the 500×500 frame, ~22.7% of
the camera image maps outside 0–12 mm travel (the left strip maps to negative motor x). The
convex-hull grid emits x-ascending from the bounding-box corner, so the dead zone comes
**first**.

A second, independent defect surfaced while designing the fix: **"Return to local control"
is a button that cannot deliver.** `take_local_control()` only sets `_raster_source =
"local"` (`raster_controller.py:1564-1577`); `raster_step` re-asserts `"remote"` on every
zmq step (`:1483`). Ownership is last-stepper-wins — nobody's decision, an artifact. This
shipped with BLACS-driven stepping in `4c826de`; it is not a regression from earlier
behaviour.

### Goals

- The display never shows a path that is not the path that will run.
- The operator can hand-drive the raster while a queue keeps firing, and take that control
  back from either the GUI or the BLACS tab.
- Nothing silently changes where the laser fires.

### Non-goals

- Moving path generation or spec-building into the controller. The spec lives in Qt
  widgets by design; `remote_arm_provider` exists precisely to bridge that.
- Any new ZMQ message type. Every change below re-means an existing message or adds a
  pseudo-connection alongside the existing specials.

---

## 2. Architecture

**The GUI owns the path. BLACS says "move to next" and nothing more.** BLACS never needs to
know who is driving; it asks the same question every shot and the GUI gives one of two
honest answers.

Two independent axes replace one overloaded flag:

| Axis | Question | Set by |
|---|---|---|
| **Raster Mode** | Is this run rastering at all? | BLACS tab checkbox (existing) |
| **Control** | Who advances the raster? | BLACS tab toggle (new) **or** the GUI, mirrored |

The two axes are genuinely independent, so **Control is never greyed out** — it describes
who may drive the motors remotely even when stepping is off:

| Raster Mode | Control | Behaviour |
|---|---|---|
| off | BLACS | No stepping. A sequence may still feed explicit coordinates to the GUI (§5) — this is remote position feeding without a pattern. |
| off | Local | No stepping and no remote position writes. Fully hand-driven. |
| on | BLACS | Remote stepping. Today's behaviour. |
| on | Local | Shots keep firing; the operator steps at the GUI; the h5 records the real site. |

Note which axis gates what: **Raster Mode gates stepping; Control gates every remote motor
command**, including the buffered coordinate write. That is what makes "Raster off +
Control BLACS" a useful state rather than a dead one.

---

## 3. Component: the display (GUI)

Three representations exist today; only the third moves motors. The fix is to make the
first two visually distinct and the third the only thing drawn as live.

| | Lives in | Drawn as | Changes when |
|---|---|---|---|
| **Pending** | GUI `_pending_pts` | dashed grey | every pattern edit |
| **Armed** | controller `_raster_path_pts` | solid green, numbered | only on Re-arm |

Changes:

1. **While armed, render the controller's armed path**, not the `_raster_preview_pts`
   cache. That cache is what froze and lied.
2. **Delete the early-return at `ui.py:1547-1548`.** Pending refreshes freely on every
   edit now that it is visually distinct from armed.
3. **Re-arm is one click, always enabled — including while Control=BLACS.** It swaps
   pending into armed. It does not advance the cursor, so it cannot desync BLACS's shot
   count, and the GUI does not need BLACS's permission to change its own path. This is
   distinct from the gating in §4, which constrains the three *BLACS-side* arm senders;
   the GUI's own Re-arm button is never gated on ownership. Consequence to accept: shots
   before and after a mid-queue Re-arm land on different patterns, and `raster_point_meta`
   is what distinguishes them in the h5.
4. **Status line** reads `armed 47 pts | pending 62 pts` when they differ, collapsing to
   `armed 47 pts` when they match.
5. **Dead-zone overlay:** shade the region where `target_to_motor` leaves `motor_bounds`,
   recomputed on `calibration_ready_signal`. This is what stops the pattern being drawn
   into the dead zone in the first place.

### Unreachable points

The arm-time reachability filter (already on `fix/raster-reachability` @ `7360298`) stays.
Its predicate is exactly the step-time gate — both do `_target_to_motor_clamped(cal, p)`
then `_within_bounds(·, motor_bounds)` — and `motor_bounds` is set once at construction
(`main_rastering.py:47`) and never mutated, so it drops only points that would have been
rejected anyway. No false positives.

Point numbering follows the **armed** path: progress reads `34/70` and means it. Original
hull indices are deliberately **not** preserved — skipping numbers would be its own lie,
and the confirm step below removes the surprise the preservation was meant to address.

**Both arming paths behave identically: drop the unreachable points, arm the rest, and say
so on the console.** No modal, no confirm gate. The confirm step is unnecessary once the
display stops lying — the armed path is drawn accurately the moment it is armed, so the
operator *sees* the reduced pattern rather than being asked about it. A modal would also
have timed out against the 10 s remote arm timeout (`raster_controller.py:546`) whenever
nobody was at the GUI.

The console message names the count and the frame, e.g.
`Raster armed: 47 of 62 points; 15 dropped, outside motor travel (0.0, 12.0, 0.0, 12.0) mm.`

The remote reply additionally carries `extra={"armed": 47, "dropped": 15}`; the worker logs
`armed`/`dropped` from the arm reply at both arm call sites (Task 5), so the drop lands in
the BLACS log too. The queue never stalls for a pattern that merely overhangs the frame.

Total-drop still refuses to arm, as today (`:1395-1401`).

---

## 4. Component: ownership (GUI + BLACS)

**Ownership changes only when a human changes it**, and exactly one driver is live at a
time:

| Control | GUI Step button | BLACS `move_to_next` |
|---|---|---|
| BLACS | disabled — a local step would desync BLACS's shot count | advances |
| Local | enabled | acknowledges, does not advance |

**One cursor, shared.** `raster_step` advances the same `_next_raster_point_locked`
regardless of caller (`raster_controller.py:1478`), so handing off mid-pattern in either
direction resumes where the other left off — hand-step to point 12, flip to BLACS, BLACS
takes point 13. Nothing resets. Either side may step through the path; the toggle only
decides which side is doing it right now.

**Handing over mid-continuous converts, at the GUI only.** BLACS drives point-by-point, so
a continuous run cannot survive the hand-off — both would advance the one cursor.
`give_remote_control()` therefore clears `_raster_continuous` under the same lock that
flips ownership (the chain stops after the in-flight point) and emits a status line naming
where the cursor sits, e.g. `Continuous run stopped - BLACS drives step-by-step from point
13/47`. The raster stays armed and the cursor is preserved, so BLACS resumes mid-path, not
at point 0. The **remote** side may never make this conversion: an `arm_raster` that would
re-mode a continuous run down to step is refused with `raster_in_continuous_mode` (§6),
the same rule `disarm_raster` already enforces. Ending or converting a continuous sweep is
a human decision, and the only human path into it is this button.


- **Delete the ownership flip in `raster_step`** — `new_source = "remote" if source ==
  "zmq" else "local"` (`raster_controller.py:1483-1484`). A zmq step must stop seizing
  control.
- **`disarm_raster` releases ownership instead of calling `stop_raster()`**
  (`raster_controller.py:643`). The handler writes `_raster_source = "local"` inline rather
  than delegating to `take_local_control()`, which returns False and touches nothing when
  no raster is active. Inlining is what makes the promotion unconditional: `None → "local"`
  lands even with nothing armed, which is what lets a fresh GUI fire in place (§5).
  Release ownership, preserve the path. Keep the
  `_remote_shots_per_step = None` reset and the `raster_in_continuous_mode` refusal
  (`:626-635`) unchanged. Destroying the path becomes the GUI Stop button's job alone.

  *This message must keep being sent.* It is the only remote path that clears
  `_raster_source`; dropping it would pin the GUI at `"remote"` forever and lock the
  operator out of the Step button (`:1469-1470`).

- **Gate all three BLACS arm senders on `Control == BLACS`**, not just `move_to_next`.
  `arm_raster`'s already-armed branch sets `_raster_source = "remote"` at `:484`, and it is
  reached from the eager sync on tick (`blacs_workers.py:183-186`), the lazy arm inside
  `transition_to_buffered` (`:243-247`), and the reconnect re-sync (`:74`). Gating one is
  not gating Local control. BLACS must not arm on the operator's behalf.

- **Restore fire-in-place when nothing is armed.** Today `raster_not_active` clears the
  armed flag and then raises (`blacs_workers.py:287-292`) — the comment says "the shot
  still fails loudly". That behaviour arrived with per-shot stepping in `42c815f`; before
  it, BLACS never touched the raster and every shot simply fired where the laser sat. Firing
  in place is the intended operation, not a degraded one. Under Control=Local with nothing
  armed, the shot **fires at the current position** and does not raise. Control=BLACS keeps
  the existing arm-from-scratch retry, which is what that path is for.

- **Status mirror:** publish `manual` on the existing `raster_mode` PUB topic when armed
  and locally owned. The tab already renders `"Raster: Manual"` (`blacs_tabs.py:398`) — it
  is dead code today. No new topic.

So the Control toggle *is* the `arm_raster` / `disarm_raster` pair, re-meant, and the
status mirror is the PUB topic that already exists.

**Ownership mirror (tab checkbox).** Ownership is mirrored tab↔GUI via a dedicated
`raster_owner` PUB value (`local`/`remote`/`none`) — `raster_mode` cannot carry this alone,
since a locally-owned continuous raster publishes `continuous`, not `manual`. The tab's
Control checkbox is driven from `raster_owner`, and every real change to it routes through
`update_raster_control` — the same call the operator's own click makes — so the widget and
the worker's ownership state can never disagree silently. Deliberate rule: an incoming
`local` unticks the box only while a raster is armed; an idle GUI publishing `local` must
not fight the operator's choice of Control=BLACS for pattern-less remote position feeding.

---

## 5. Component: the per-shot round trip (BLACS)

In Local mode BLACS still asks every shot. Suppressing the round trip is not an option:
`_raster_meta` deliberately survives a shot group (`blacs_workers.py:59-60`), so a
suppressed Local shot would be stamped with the last BLACS-driven point.

**No new message and no new pseudo-connection.** BLACS keeps sending plain `move_to_next`;
the GUI decides what it means from the ownership it already holds:

- Control=BLACS → advance, return the new point's meta (today's behaviour).
- Control=Local → **do not advance**, return SUCCESS with the *current* point's meta.

BLACS never needs to signal intent, which is what "BLACS stays dumb" means. This also
sidesteps both wire constraints the audit raised, rather than working around them:
`program_value` hardcodes `args = {"wait_for_lock": ...}`
(`RemoteControl/blacs_workers.py:361`) with no extension point — irrelevant, because we
send no argument; and `extra.advanced=False` would be filtered out by `RASTER_META_KEYS`
(`blacs_workers.py:11-12`, `:294`) — irrelevant, because BLACS does not need the flag. It
already knows which mode it is in: it owns the toggle.

Control provenance in the shot record is stamped BLACS-side from that local knowledge —
`raster_control: "blacs"|"local"` written in `post_experiment`. Optional; drop it if the
`target_xy` already in `raster_point_meta` is enough provenance for analysis.
- **Bypass the group counter in Local mode.** `_advance_raster` only queries on the first
  shot of each group (`:236`). With `shots_per_step=3` and Control=Local, the operator can
  step at the GUI between shots 2 and 3 and those shots get stamped with a site the laser
  has left. Query every shot in Local mode; keep mutating `_shots_since_step` so the group
  phase survives a flip back to BLACS mid-queue.
- **Guard the never-stepped case.** `raster_point_meta` reads `_raster_index - 1`
  (`raster_controller.py:1534`), which is `-1` on a freshly armed raster. Report that
  honestly: `point_index: -1` with `target_xy` set to the laser's actual cached
  position. Two alternatives were rejected: a typed `raster_not_stepped` error would sticky-pause
  the queue on the first shot of every hand-driven run; fabricating point 0's
  coordinates would be plausible-and-wrong in the h5.

### Explicit sequence coordinates

The buffered coordinate write (`blacs_workers.py:320-349`) is currently ungated. It is
gated on **Control**, not Raster Mode:

- **Control=BLACS** → honoured, whether or not Raster Mode is on. Raster off + Control
  BLACS is exactly the "remote position feeding, no pattern" state (§2).
- **Control=Local** → **raise**. Not obeyed, not silently suppressed. A sequence naming an
  explicit position is asserting remote intent, which contradicts Local; the operator flips
  Control to BLACS from either end and resumes. Consequence, stated plainly: this pauses the
  queue stickily (§6).

---

## 6. Error semantics

Per the fork's yellow-vs-red rules: a raise inside `transition_to_buffered` is caught by the
mainloop (`tab_base_classes.py:876-880`) and appended to `error_message`. That is **yellow**
— the tab stays alive; `ICON_FATAL_ERROR` requires `state == 'fatal error'` (`:411-412`).

But the queue manager polls `error_message` alone, including in
`transition_device_to_buffered` (`experiment_queue.py:497-498`), which then returns False
for **every subsequent shot**. `_error` clears only on the tab's ✕ or a restart
(`tab_base_classes.py:747`). So any raise below pauses the queue until the banner is
dismissed:

| Condition | Outcome |
|---|---|
| Sequence coords while Control=Local | raise → sticky queue pause (intended) |
| Nothing armed, Control=Local | **fires in place, no raise** — reverses `42c815f`, see §4 |
| Nothing armed, Control=BLACS | existing arm-from-scratch retry (`blacs_workers.py:237-247`) |
| Continuous raster running, Control=Local | **SUCCESS ack carrying the sweep's last commanded point, no raise** — the operator is driving; a raise here would pause the queue every shot of a hand-run sweep |
| Continuous raster, Control=BLACS | `raster_in_continuous_mode` → raise → sticky pause — a free-running sweep cannot answer a request for per-shot coordinates |
| Remote `arm_raster` (step) while a continuous run is active | `raster_in_continuous_mode` — refused, sweep untouched. Only the GUI's "Give to BLACS" converts a continuous run (§4); it clears the flag first, so this never fires on the sanctioned hand-over. From the tab's eager arm the refusal is swallowed and logged (`_sync_raster_mode_to_gui` never raises), leaving `_raster_armed` False and the sweep running |
| Remote arm, **all** points unreachable | `no_raster_configured` → raise → sticky pause |
| Remote arm, **some** points unreachable | SUCCESS + `extra.dropped` — **no** pause |
| Eager arm on tick fails | swallowed and logged by design (`blacs_workers.py:183-193`); never raises |

---

## 7. The Control toggle's mode mask

`@define_state(MODE_MANUAL, True)` is wrong for this control. `queue_state_indefinitely=True`
does **not** drop a slot fired in a disallowed mode — `StateQueue.check_for_next_item` only
deletes in the `elif not queue_state_indefinitely` branch
(`tab_base_classes.py:157-160`). The slot **parks until queue end**: the widget would read
Local while the worker kept stepping for the rest of the queue. Exactly backwards for a
control reached for *because* shots are running.

Required:

```python
@define_state(MODE_MANUAL | MODE_TRANSITION_TO_BUFFERED | MODE_BUFFERED | MODE_POST_EXP, True)
```

`MODE_POST_EXP` (=32) is the between-queued-shots window; omit it and the flip cannot land
between shots. `Raster Mode` and `shots_per_step` stay on `MODE_MANUAL` — their tooltip
already promises queue-end semantics (`blacs_tabs.py:196-200`).

**Minimal widget: one checkbox that both shows and sets.** No separate read-only indicator.
The PUB-driven `setChecked` is deliberately **not** wrapped in `blockSignals`: the mirror
fires the real `toggled` slot, so every change to the widget — operator click or incoming
status — routes through `update_raster_control` and the worker's ownership can never drift
from what the box shows. The echo terminates on its own, because `setChecked` emits
`toggled` only on an actual change of value: the slot's own write finds the box already in
that state and nothing re-fires. Suppressing the slot is precisely how the widget and the
worker diverged in the first place — the incident this branch exists to fix — so the
sandwich is the wrong reflex here even though it is the right one for
`restore_save_data` below.

Widget wiring follows the tab's existing pattern exactly: one key added to `get_save_data`
(`blacs_tabs.py:424-427`), one `blockSignals`-wrapped restore (`:440-448`, mandatory —
`restore_save_data` runs before `initialise_workers`, so `primary_worker` is still `None`),
one key in the `initialise_workers` workerargs (`:348`), normalized worker-side with
`getattr` in `_init_raster_state` (`blacs_workers.py:52-53`). Sibling widget reads inside a
`define_state` body must go through `inmain` (`:373`) — never `qtlock`. With three
interacting controls, hold the state in plain Python attrs on the tab, updated on the GUI
thread, and read those instead of `inmain`-reading two siblings from three slots.

---

## 8. Testing

Three new asserts in the existing `tests/test_raster_pathmodel.py` — one of the two
camera-safe files, alongside `tests/test_zmq_v2_protocol.py` (the rest import `ui.py`,
which opens the uEye camera and hangs while the GUI runs). Run both before any commit.
Everything else here is either already covered by the existing 48 tests or is a one-liner
whose failure is obvious on first use.

| Assert | Pins |
|---|---|
| a zmq step leaves `_raster_source` unchanged | the deleted flip at `:1483` — this *is* the bug |
| `disarm_raster` releases ownership and **preserves** `_raster_path_pts` | the re-meaning in §4 |
| `move_to_next` while Control=Local returns meta without advancing `_raster_index` | §5 |

### Runtime check (operator, after deploy)

Two things that would otherwise fail silently:

1. **Flip Control to Local mid-queue** → the *next* shot honours it. If it only takes
   effect at queue end, the mode mask (§7) is wrong.
2. **Untick Raster Mode** → the armed path survives and the indicator reads `Control:
   Local` (because `disarm_raster` now releases rather than destroys).

The rest — GUI Step in Local mode, the sequence-coords raise, the h5 site — announce
themselves the first time you use them.

---

## 9. Scope and rollout

| Change | Repo | Restart |
|---|---|---|
| Display split, Re-arm, overlay, ownership, `move_to_next` Local-mode handling, `disarm_raster` semantics | `GUIs/rastering` | rastering GUI only |
| Control toggle, three-sender gating, Local-mode round trip, `raster_control` h5 stamp (optional, per §5) | parent `userlib/user_devices/RasteringDevice/` | BLACS |

**No connection-table change and no recompile.** `arm_raster`, `move_to_next`,
`disarm_raster`, `shots_per_step` are pseudo-connections dispatched inside `_handle_program`
(`raster_controller.py:366-367`), not `RemoteAnalogOut` children; the tab reads only
`mock/host/reqrep_port/pubsub_port` from device properties (`blacs_tabs.py:86-89`).

**Keep every BLACS-side change inside `RasteringDevice/`.** `RemoteControlTab` /
`RemoteControlWorker` are shared with `LaserLockDevice` and `BigSkyHub`; the
pseudo-connection in §5 exists specifically to avoid touching them.

**Deploy order: GUI first, then BLACS.** Deploying BLACS first, then flipping Control to
Local, sends `disarm_raster` to an old GUI that still implements it as `stop_raster()` —
destroying the armed path. After that, the gated worker never re-arms, and every shot
sticky-pauses the queue on `raster_not_active`. GUI-first avoids this: by the time BLACS
can send `disarm_raster`, the GUI already answers it by releasing ownership (§4).

**Branching.** The parent repo is currently on `master`, which the operator runs between
shots — BLACS-side work goes on its own branch/worktree, never in place. GUI work continues
on `fix/raster-reachability` (worktree `GUIs/rastering-hull-reach`), which also still needs
merging to the rastering `main`.
