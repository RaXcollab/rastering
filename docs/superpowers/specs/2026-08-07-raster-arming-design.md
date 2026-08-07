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

Three honest states:

| Raster Mode | Control | Behaviour |
|---|---|---|
| off | — | BLACS never calls `move_to_next`. Control greyed. |
| on | BLACS | Remote stepping. Today's behaviour. |
| on | Local | Shots keep firing; the operator steps at the GUI; the h5 records the real site. A sequence that programs explicit coords **raises** (§5). |

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

Two arming paths, deliberately different:

- **Local arm** (Auto Raster / Re-arm) with unreachable points → show the **reduced**
  preview and wait for the operator to confirm before arming.
- **Remote arm** (`arm_raster` from BLACS, nobody at the GUI) → arm the reachable points
  and report the drop in the reply: `extra={"armed": 47, "dropped": 15}`. The queue never
  stalls for a pattern that merely overhangs the frame. A modal here would just time out
  against the 10 s arm timeout (`raster_controller.py:546`).

Total-drop still refuses to arm, as today (`:1395-1401`).

---

## 4. Component: ownership (GUI + BLACS)

**Ownership changes only when a human changes it.**

- **Delete the ownership flip in `raster_step`** — `new_source = "remote" if source ==
  "zmq" else "local"` (`raster_controller.py:1483-1484`). A zmq step must stop seizing
  control.
- **`disarm_raster` calls `take_local_control()` instead of `stop_raster()`**
  (`raster_controller.py:643`). Release ownership, preserve the path. Keep the
  `_remote_shots_per_step = None` reset and the `raster_in_continuous_mode` refusal
  (`:626-635`) unchanged. Destroying the path becomes the GUI Stop button's job alone.

  *This message must keep being sent.* It is the only remote path that clears
  `_raster_source`; dropping it would pin the GUI at `"remote"` forever and lock the
  operator out of the Step button (`:1469-1470`).

- **Gate all three BLACS arm senders on `Control == BLACS`**, not just `move_to_next`.
  `arm_raster`'s already-armed branch sets `_raster_source = "remote"` at `:484`, and it is
  reached from the eager sync on tick (`blacs_workers.py:183-186`), the lazy arm inside
  `transition_to_buffered` (`:243-247`), and the reconnect re-sync (`:74`). Gating one is
  not gating Local control. In Local mode with nothing armed, let the GUI answer
  `raster_not_active` and fail the shot loudly — BLACS must not arm on the operator's
  behalf.

- **Status mirror:** publish `manual` on the existing `raster_mode` PUB topic when armed
  and locally owned. The tab already renders `"Raster: Manual"` (`blacs_tabs.py:398`) — it
  is dead code today. No new topic.

So the Control toggle *is* the `arm_raster` / `disarm_raster` pair, re-meant, and the
status mirror is the PUB topic that already exists.

---

## 5. Component: the per-shot round trip (BLACS)

In Local mode BLACS still asks every shot. Suppressing the round trip is not an option:
`_raster_meta` deliberately survives a shot group (`blacs_workers.py:59-60`), so a
suppressed Local shot would be stamped with the last BLACS-driven point.

- **Do not overload `move_to_next` with an `advance` argument.** `program_value` hardcodes
  `args = {"wait_for_lock": bool(wait_for_lock)}` (`RemoteControl/blacs_workers.py:361`)
  with no extension point; extending it would change the base class shared by Laser Lock
  and BigSky.
- **Add a `raster_current_point` pseudo-connection** next to the existing specials
  (`raster_controller.py:366-367`), returning `encode_reply(status="SUCCESS",
  extra=self._outer.raster_point_meta())`. `extra` merges flat into the reply
  (`external_gui_lib/zmq_v2.py:222-223`), so the existing `RASTER_META_KEYS` filter picks
  it up unchanged. In Local mode BLACS sends this instead of `move_to_next`.
- **`extra.advanced=False` is not viable** — `blacs_workers.py:294` filters the reply
  against a fixed `RASTER_META_KEYS` tuple (`:11-12`), so it is silently dropped; and it
  would be wrong for `shots_per_step > 1`, where the whole group inherits shot 1's flag.
  Stamp control provenance BLACS-side instead: `raster_control: "blacs"|"local"` written in
  `post_experiment` from local knowledge.
- **Bypass the group counter in Local mode.** `_advance_raster` only queries on the first
  shot of each group (`:236`). With `shots_per_step=3` and Control=Local, the operator can
  step at the GUI between shots 2 and 3 and those shots get stamped with a site the laser
  has left. Query every shot in Local mode; keep mutating `_shots_since_step` so the group
  phase survives a flip back to BLACS mid-queue.
- **Guard the never-stepped case.** `raster_point_meta` reads `_raster_index - 1`
  (`raster_controller.py:1534`); on a freshly armed raster that is `-1`, yielding
  `point_index: -1` and no `target_xy` — a silently bogus h5 record. Return a typed
  `raster_not_stepped` error instead.

### Explicit sequence coordinates

The buffered coordinate write (`blacs_workers.py:320-349`) is currently ungated. Under
Control=Local it must **raise**, not obey and not silently suppress. A sequence naming an
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
| `move_to_next` hits `raster_not_active` (operator pressed Stop) | raise → sticky pause. Loud and correct; in Local mode there is no re-arm retry to save the operator. |
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

Camera-safe unit tests only — `tests/test_raster_pathmodel.py` is the sole file that does
not import `ui.py` (which opens the uEye camera and hangs while the GUI runs).

| Test | Pins |
|---|---|
| zmq step does not change `_raster_source` | the deleted flip at `:1483` |
| `disarm_raster` releases ownership and **preserves** `_raster_path_pts` | §4 |
| `arm_raster` while Control=Local does not set `_raster_source = "remote"` | the `:484` seizure |
| `raster_current_point` returns current meta without advancing `_raster_index` | §5 |
| `raster_point_meta` on a never-stepped raster returns `raster_not_stepped` | the `-1` bogus record |
| remote arm with partial drop → SUCCESS + `armed`/`dropped` counts | §3 |
| remote arm with total drop → refuses | regression guard on `:1395-1401` |
| progress numerator/denominator follow the filtered path | `34/70` honesty |

BLACS-side `define_state` mask and widget persistence are verified by the runtime checklist
below, not by unit tests.

### Runtime verification (operator, after deploy)

1. Tick Raster Mode with Control=BLACS → GUI arms, path visible.
2. Flip Control to Local **mid-queue** → the *next* shot honours it. If it only takes effect
   at queue end, the mode mask is wrong.
3. Press GUI Step in Local mode → accepted.
4. Untick Raster Mode → the armed path **survives**, indicator reads `Control: --`.
5. Run a sequence with explicit `laser_raster_x_coord` while Control=Local → shot raises,
   queue pauses.
6. Check `/data/RasteringGUI/raster` attrs for a Local-mode shot against where the laser
   actually was.

---

## 9. Scope and rollout

| Change | Repo | Restart |
|---|---|---|
| Display split, Re-arm, overlay, ownership, `raster_current_point`, `disarm_raster` semantics | `GUIs/rastering` | rastering GUI only |
| Control toggle, three-sender gating, Local-mode round trip, `raster_control` h5 stamp | parent `userlib/user_devices/RasteringDevice/` | BLACS |

**No connection-table change and no recompile.** `arm_raster`, `move_to_next`,
`disarm_raster`, `shots_per_step` are pseudo-connections dispatched inside `_handle_program`
(`raster_controller.py:366-367`), not `RemoteAnalogOut` children; the tab reads only
`mock/host/reqrep_port/pubsub_port` from device properties (`blacs_tabs.py:86-89`).

**Keep every BLACS-side change inside `RasteringDevice/`.** `RemoteControlTab` /
`RemoteControlWorker` are shared with `LaserLockDevice` and `BigSkyHub`; the
pseudo-connection in §5 exists specifically to avoid touching them.

**Deploy order: GUI first, then BLACS.** BLACS only sends `raster_current_point` in Local
mode, and an old GUI would answer `unknown_connection` (`:650-651`) → strict raise → queue
pause. The `disarm_raster` semantic change is symmetrical and order-free.

**Branching.** The parent repo is currently on `master`, which the operator runs between
shots — BLACS-side work goes on its own branch/worktree, never in place. GUI work continues
on `fix/raster-reachability` (worktree `GUIs/rastering-hull-reach`), which also still needs
merging to the rastering `main`.
