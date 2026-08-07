# Raster Arming Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the rastering GUI's display always show the path that will actually run, and give "local control" real authority so the operator can hand-drive the raster while a BLACS queue keeps firing.

**Architecture:** Two overloaded concepts get split into two independent flags. `Raster Mode` (BLACS tab) gates *stepping*; `Control` (BLACS tab + GUI, mirrored) gates *every remote motor command*. Ownership stops being an artifact of who stepped last and becomes a value only a human sets. On screen, the controller's armed path is the only thing drawn as live; edits render as a separate dashed "pending" path until an explicit Re-arm.

**Tech Stack:** Python 3.11, PyQt5, pyqtgraph, pyzmq (v2 RemoteControl protocol), pytest. GUI runs in conda env `rastering`; BLACS side runs in conda env `labscript`.

**Spec:** `docs/superpowers/specs/2026-08-07-raster-arming-design.md`

## Global Constraints

- GUI conda env is `rastering`, NOT `labscript`. Run tests with:
  `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q`
- **Only `tests/test_raster_pathmodel.py` is camera-safe.** `test_command_queue.py` and `test_raster_goto_handlers.py` import `ui.py`, which opens the uEye camera and HANGS while the GUI is running. Never run the whole `tests/` directory.
- Use `python -m py_compile <file>` to syntax-check `ui.py` — never import it.
- **Never stage, commit, or restore `calibration_data.json`.** It is tracked-but-always-dirty operator data. Stage only files named explicitly in each task.
- `motor_bounds` is motor-space mm; `target_bounds` is pixel/target space. Never mix.
- No new ZMQ message types. Every change re-means an existing message.
- Tasks 1–6 land in the rastering repo (worktree `GUIs/rastering-hull-reach`, branch `fix/raster-reachability`). Tasks 7–8 land in the parent repo under `userlib/user_devices/RasteringDevice/` on a NEW branch — the parent is on `master`, which the operator runs between shots. Never commit to parent `master`.
- **Deploy order is GUI first, then BLACS** (§9 of the spec).

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `raster_controller.py` | ownership flag, ZMQ handlers, arm-time filter, PUB status | 1, 2, 3, 5 |
| `ui.py` | armed-vs-pending rendering, Re-arm button, dead-zone overlay | 4, 6 |
| `tests/test_raster_pathmodel.py` | the three camera-safe asserts | 1, 3 |
| `RasteringDevice/blacs_tabs.py` (parent) | Control checkbox, persistence, mode mask | 7 |
| `RasteringDevice/blacs_workers.py` (parent) | gate arm senders + buffered write, fire-in-place | 8 |

---

## Task 1: Ownership becomes a human decision

Today `raster_step` sets `_raster_source` on every step (`raster_controller.py:1483-1484`), so a zmq step silently seizes the raster and "Return to local control" cannot hold. And `disarm_raster` destroys the armed path when it should only release it.

**Files:**
- Modify: `raster_controller.py:1465-1487` (the `raster_step` ownership flip)
- Modify: `raster_controller.py:1579-1594` (`stop_raster` clearing ownership)
- Modify: `raster_controller.py:637-644` (`disarm_raster` handler)
- Test: `tests/test_raster_pathmodel.py`

**Interfaces:**
- Produces: `_raster_source` is now a persistent ownership flag with values `"remote"` / `"local"` / `None`, written ONLY by `start_raster`, `arm_raster`, `disarm_raster`, and `take_local_control`. It is no longer cleared by `stop_raster` and no longer written by `raster_step`. Task 3 and Task 8 both read it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_raster_pathmodel.py`, after `test_raster_step_empty_path_finishes`:

```python
def test_zmq_step_does_not_seize_ownership():
    """A zmq step must NOT flip ownership to 'remote'. Last-stepper-wins made
    _raster_source an artifact of who called last, which is how BLACS silently
    took the raster back from the operator (2026-08-07 incident)."""
    sc = _step_self([(1.0, 2.0), (3.0, 4.0)], active=True)
    sc._raster_source = "local"
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._raster_source == "local", "zmq step must not change ownership"


def test_stop_raster_preserves_ownership():
    """stop_raster tears down the PATH, not the ownership flag. Ownership must
    survive so a move_to_next arriving with nothing armed can still tell that
    the operator holds control and fire in place (Task 3)."""
    sc = _step_self([(1.0, 2.0)], active=True)
    sc._raster_source = "local"
    SystemController.stop_raster(sc)
    assert sc._raster_path_pts == []
    assert sc._raster_active is False
    assert sc._raster_source == "local"
```

`_step_self` already exists in this file. If it does not set `_raster_source`, add `_raster_source=None` to its `SimpleNamespace` and a `_flush_raster_log=mock.Mock()` / `selection_changed_signal=mock.Mock()` attribute if `stop_raster` needs them — read `_step_self` first and extend it minimally rather than writing a new helper.

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```
& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q -k "seize or preserves_ownership"
```
Expected: FAIL — `assert 'remote' == 'local'` on the first, `assert None == 'local'` on the second.

- [ ] **Step 3: Remove the ownership flip from `raster_step`**

In `raster_controller.py`, replace lines 1480-1487:

```python
            if stepping:
                # Whoever advances the raster owns it from here: the UI's Step
                # button (source "ui") or BLACS's move_to_next (source "zmq").
                new_source = "remote" if source == "zmq" else "local"
                self._raster_source = new_source

        if stepping:
            self.raster_source_signal.emit(new_source)
```

with:

```python
        # Ownership is deliberately NOT touched here. Last-stepper-wins made
        # _raster_source an artifact of whoever called last, so every BLACS
        # step silently seized the raster and "Return to local control" could
        # never hold (2026-08-07 incident). Ownership now changes only when a
        # human changes it: start_raster, arm_raster, disarm_raster, or
        # take_local_control.
```

- [ ] **Step 4: Stop `stop_raster` clearing ownership**

In `raster_controller.py`, inside `stop_raster`, delete the line `self._raster_source = None` and the `self.raster_source_signal.emit(None)` that follows in the `if was_active:` block. Add above the remaining state writes:

```python
            # _raster_source is NOT cleared: it is the ownership flag, not path
            # state. It must outlive the path so a move_to_next arriving with
            # nothing armed can still tell that the operator holds control and
            # fire in place rather than failing the shot.
```

- [ ] **Step 5: Make `disarm_raster` release instead of destroy**

In `raster_controller.py`, in the `disarm_raster` handler, replace:

```python
            if active:
                # stop_raster is safe from this thread: state writes are under
                # _state_lock, the emits are cross-thread-queued Qt signals, and
                # the log flush is plain file IO -- no widgets, no motor DLL.
                # (The was_active guard inside it also makes a race with the
                # GUI's own Stop button a single-flush.)
                self._outer.stop_raster()
                self._outer.status_signal.emit("ZMQ: raster disarmed by BLACS.")
```

with:

```python
            # RELEASE, do not destroy. BLACS unticking Raster Mode means "I
            # stop driving", not "throw the operator's pattern away" -- one
            # checkbox was doing two jobs. Stop at the GUI is the only path
            # that destroys an armed raster.
            with self._outer._state_lock:
                self._outer._raster_source = "local"
            self._outer.raster_source_signal.emit("local")
            if active:
                self._outer.status_signal.emit(
                    "ZMQ: BLACS released the raster; local control.")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```
& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q
```
Expected: PASS, 50 tests (48 existing + 2 new). If any pre-existing test asserted the old last-stepper-wins behaviour, it is now testing a bug — update it to assert the new invariant and note that in the commit message.

- [ ] **Step 7: Commit**

```bash
git commit --only raster_controller.py tests/test_raster_pathmodel.py -m "fix(raster): ownership changes only when a human changes it

raster_step set _raster_source on every step, so a zmq step silently seized
the raster and 'Return to local control' could never hold -- it flipped back
on BLACS's next move_to_next. Ownership is now written only by start_raster,
arm_raster, disarm_raster and take_local_control.

stop_raster no longer clears the flag: ownership is not path state, and it
must outlive the path so a move_to_next with nothing armed can tell the
operator holds control.

disarm_raster releases ownership instead of calling stop_raster. Unticking
Raster Mode in BLACS meant 'stop driving' AND 'destroy the pattern'; only
the GUI Stop button destroys now."
```

---

## Task 2: Publish `manual` so both screens agree

The BLACS tab already renders `"Raster: Manual"` (`blacs_tabs.py:398`) but the GUI never publishes it — dead code since the topic shipped.

**Files:**
- Modify: `raster_controller.py:2393-2403` (the PUB status block)

**Interfaces:**
- Produces: `raster_mode` PUB topic gains a fourth value, `"manual"`, meaning "armed but the operator holds control". Task 7 consumes it to drive the tab's Control checkbox.

- [ ] **Step 1: Publish `manual` when armed and locally owned**

In `raster_controller.py`, replace:

```python
                    with self._state_lock:
                        active = self._raster_active
                        continuous = self._raster_continuous
                        cal = self.calibration

                    if not active:
                        publish("raster_mode", "idle")
                    elif continuous:
                        publish("raster_mode", "continuous")
                    else:
                        publish("raster_mode", "step")
```

with:

```python
                    with self._state_lock:
                        active = self._raster_active
                        continuous = self._raster_continuous
                        source = self._raster_source
                        cal = self.calibration

                    # "manual" = armed, but the operator holds control, so
                    # BLACS's move_to_next acknowledges without advancing.
                    # The BLACS tab has rendered this value since the topic
                    # shipped (blacs_tabs.py:398); nothing ever sent it.
                    if not active:
                        publish("raster_mode", "idle")
                    elif continuous:
                        publish("raster_mode", "continuous")
                    elif source == "local":
                        publish("raster_mode", "manual")
                    else:
                        publish("raster_mode", "step")
```

- [ ] **Step 2: Syntax-check**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m py_compile raster_controller.py`
Expected: no output, exit 0.

- [ ] **Step 3: Run the suite to confirm nothing regressed**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q`
Expected: PASS, 50 tests.

- [ ] **Step 4: Commit**

```bash
git commit --only raster_controller.py -m "feat(raster): publish 'manual' when the operator holds an armed raster

The BLACS tab has rendered 'Raster: Manual' since the raster_mode topic
shipped (blacs_tabs.py:398) but nothing ever published it. Both screens now
agree on who is driving, over a topic that already exists."
```

---

## Task 3: `move_to_next` under local control acknowledges without advancing

**Files:**
- Modify: `raster_controller.py:565-597` (the `move_to_next` handler)
- Test: `tests/test_raster_pathmodel.py`

**Interfaces:**
- Consumes: `_raster_source` from Task 1.
- Produces: `move_to_next` returns SUCCESS in two new cases — nothing armed under local control (`extra={"in_place": True}`), and armed under local control (`extra=raster_point_meta()` for the CURRENT point, cursor unmoved). Task 8 relies on neither raising.

- [ ] **Step 1: Write the failing test**

```python
def test_move_to_next_under_local_control_does_not_advance():
    """BLACS asking for the next point while the operator holds control must
    acknowledge without moving the cursor -- the shot fires where the operator
    put the laser, and the queue keeps running."""
    sc = _step_self([(1.0, 2.0), (3.0, 4.0)], active=True)
    sc._raster_source = "local"
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._raster_index == 0, "cursor must not advance under local control"
    assert sc._q.qsize() == 0, "no motor command may be enqueued"
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q -k "under_local_control"
```
Expected: FAIL — `assert 1 == 0`, the cursor advanced.

- [ ] **Step 3: Add the local-control guard to `raster_step`**

In `raster_controller.py`, inside `raster_step`'s locked block, replace:

```python
            refused_remote = (active and source != "zmq"
                              and self._raster_source == "remote")
            stepping = active and not continuous and not refused_remote
```

with:

```python
            refused_remote = (active and source != "zmq"
                              and self._raster_source == "remote")
            # Mirror image: BLACS must not advance a raster the operator holds.
            # This is NOT an error -- the shot still fires, at whatever point
            # the operator last stepped to. The handler turns it into a SUCCESS
            # reply carrying the current point's meta.
            held_by_operator = (active and source == "zmq"
                                and self._raster_source == "local")
            stepping = (active and not continuous
                        and not refused_remote and not held_by_operator)
```

Then, in the early-return block below, add after the `refused_remote` branch:

```python
        if held_by_operator:
            # Silent by design: the queue is running and this is the normal
            # state while the operator hand-drives. The handler replies SUCCESS.
            return None
```

- [ ] **Step 4: Make the handler distinguish the three None cases**

In `raster_controller.py`, replace the `move_to_next` handler body:

```python
        if connection == "move_to_next":
            with self._outer._state_lock:
                active = self._outer._raster_active
                continuous = self._outer._raster_continuous
            if not active:
                return self._err(
                    request_id=request_id, code="raster_not_active",
                    message="raster not active",
                )
```

with:

```python
        if connection == "move_to_next":
            with self._outer._state_lock:
                active = self._outer._raster_active
                continuous = self._outer._raster_continuous
                held_by_operator = self._outer._raster_source == "local"
            if not active:
                if held_by_operator:
                    # Fire in place. Nothing is armed and the operator holds
                    # control, so the shot belongs wherever the laser already
                    # is. Per-shot stepping (42c815f) turned this into a hard
                    # failure; before it, every shot fired in place. Firing in
                    # place is intended operation, not a degraded mode.
                    return encode_reply(
                        status="SUCCESS", request_id=request_id,
                        extra={"in_place": True},
                    )
                return self._err(
                    request_id=request_id, code="raster_not_active",
                    message="raster not active",
                )
```

And replace the `res is None` branch:

```python
            if res is None:
                return encode_reply(
                    status="SUCCESS", request_id=request_id,
                    extra={"finished": True},
                )
```

with:

```python
            if res is None:
                if held_by_operator:
                    # Armed, but the operator is driving: acknowledge with the
                    # CURRENT point so the shot h5 records where the laser
                    # actually is, without moving the cursor BLACS is not
                    # driving. raster_point_meta(None) reads the cursor.
                    return encode_reply(
                        status="SUCCESS", request_id=request_id,
                        extra=self._outer.raster_point_meta(),
                    )
                # Iterator end -> SUCCESS + finished=True (not a status enum).
                return encode_reply(
                    status="SUCCESS", request_id=request_id,
                    extra={"finished": True},
                )
```

- [ ] **Step 5: Guard the never-stepped case in `raster_point_meta`**

`raster_point_meta` reads `_raster_index - 1` (`raster_controller.py:1534`); on a freshly armed, never-stepped raster that is `-1`, producing `point_index: -1` and no `target_xy` — a silently bogus h5 record, now reachable on every local-control shot. In `raster_point_meta`, after `pt = self._raster_path_pts[i] if 0 <= i < total else None`, the existing code already leaves `target_xy` absent when `pt` is None. Change the meta dict to clamp instead:

```python
        meta: Dict[str, Any] = {
            "point_index": i,
            "path_len": total,
            "frame": "pixel" if cal is not None else "motor",
        }
        if i < 0:
            # Armed but never stepped: report the point the raster WILL fire
            # at (point 0) rather than -1 with no coordinates, which would
            # land a bogus record in the shot h5.
            meta["point_index"] = 0
            pt = self._raster_path_pts[0] if total else None
```

Place this block immediately after `meta` is built and before `xy = (res.target_xy ...)`.

- [ ] **Step 6: Run the full suite**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q`
Expected: PASS, 51 tests.

- [ ] **Step 7: Commit**

```bash
git commit --only raster_controller.py tests/test_raster_pathmodel.py -m "feat(raster): local control acknowledges BLACS steps without advancing

BLACS stays dumb -- it sends plain move_to_next every shot and never asks who
is driving. The GUI answers from the ownership it already holds: advance and
return the new point under BLACS control, or acknowledge with the CURRENT
point's meta under local control, cursor unmoved.

With nothing armed under local control the reply is SUCCESS + in_place, so
the shot fires where the laser sits. Per-shot stepping (42c815f, parent repo)
had replaced that with a hard failure; before it every shot fired in place.

Also clamps raster_point_meta's point_index on a never-stepped raster, which
local control makes reachable every shot -- it previously emitted -1 with no
target_xy straight into the shot h5."
```

---

## Task 4: Armed vs pending — the display stops lying

The path preview freezes while armed (`ui.py:1547-1548`) but hull vertices keep updating (`ui.py:328-333`), so the operator reads coordinates off a path that is not the one running. This is the root cause of the 2026-08-07 incident.

**Files:**
- Modify: `ui.py:1538-1552` (`_on_raster_param_changed`)
- Modify: `ui.py:2104-2119` (`_refresh_raster_scatter`)
- Modify: `ui.py` — add `armed_path_points()` accessor call sites and the Re-arm button
- Modify: `raster_controller.py` — add an `armed_path_points()` accessor

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SystemController.armed_path_points() -> List[Tuple[float, float]]` returning a copy of `_raster_path_pts` under `_state_lock`. Task 6 does not use it; nothing else depends on this task.

- [ ] **Step 1: Add the armed-path accessor**

In `raster_controller.py`, next to `_raster_progress_text`:

```python
    def armed_path_points(self) -> List[TargetXY]:
        """A copy of the armed path, for the UI to draw. The UI must render
        THIS, never its own preview cache: the cache freezes while armed and
        was what made the display disagree with the running path (2026-08-07)."""
        with self._state_lock:
            return list(self._raster_path_pts)
```

- [ ] **Step 2: Draw the armed path while armed**

In `ui.py`, replace `_refresh_raster_scatter`'s source selection:

```python
        if not self._raster_preview_pts:
            self.raster_scatter.clear()
            return
        if self.show_all_raster_points_checkbox.isChecked():
            pts = self._raster_preview_pts
        else:
            n = int(self.raster_point_display_count.value())
            pts = self._raster_preview_pts[-n:] if n > 0 else []
```

with:

```python
        # While armed, the ARMED path is the truth -- never the preview cache,
        # which freezes on arm and is what made the screen disagree with the
        # running path (2026-08-07 incident).
        if getattr(self, "_raster_active_ui", False):
            source_pts = self.controller.armed_path_points()
        else:
            source_pts = self._raster_preview_pts
        if not source_pts:
            self.raster_scatter.clear()
            return
        if self.show_all_raster_points_checkbox.isChecked():
            pts = source_pts
        else:
            n = int(self.raster_point_display_count.value())
            pts = source_pts[-n:] if n > 0 else []
```

- [ ] **Step 3: Unfreeze the pending preview**

In `ui.py`, replace `_on_raster_param_changed`'s early return:

```python
        if getattr(self, "_raster_active_ui", False):
            return
        if not self._raster_preview_pts:
            return
        self._clear_raster_overlay()
        self._render_preview(quiet=True)
```

with:

```python
        # No longer returns early while armed. The armed path is drawn from the
        # controller (see _refresh_raster_scatter), so a live pending preview
        # can no longer be mistaken for it -- freezing it was the lie.
        if not self._raster_preview_pts:
            return
        self._clear_raster_overlay()
        self._render_preview(quiet=True)
        self._update_armed_pending_status()
```

- [ ] **Step 4: Add the status line**

In `ui.py`, next to `_update_step_mode_ui`:

```python
    def _update_armed_pending_status(self) -> None:
        """Say plainly when the armed path and the on-screen pattern differ.
        Silent when they match -- the operator only needs telling when the
        thing that will run is not the thing they are looking at."""
        if not getattr(self, "_raster_active_ui", False):
            return
        armed = len(self.controller.armed_path_points())
        pending = len(self._raster_preview_pts)
        if pending and pending != armed:
            self._log(f"armed {armed} pts | pending {pending} pts "
                      f"-- press Re-arm to run the pattern on screen")
```

- [ ] **Step 5: Add the Re-arm button**

In `ui.py`, in the block that creates `raster_remote_arm_button` (around `:749`), add after it:

```python
        if not hasattr(self, "raster_rearm_button"):
            self.raster_rearm_button = QtWidgets.QPushButton("Re-arm from pending")
            self.raster_rearm_button.setToolTip(
                "Replace the armed path with the pattern currently on screen.\n"
                "Does not move the motors and does not advance the cursor, so it\n"
                "cannot desync BLACS's shot count.")
            _place(self.raster_rearm_button, 3, 0, 2)
            self.raster_rearm_button.clicked.connect(self._on_rearm_clicked)
```

- [ ] **Step 6: Let Re-arm through `_start_raster`'s ownership gate, then add the handler**

`_start_raster` refuses while BLACS owns the raster (`ui.py:1589-1592`). That gate exists to stop a local *Start* from re-arming behind BLACS's back — but Re-arm is exactly that operation, deliberately. Widen the signature first:

```python
    def _start_raster(self, *, source: str = "local", rearm: bool = False) -> None:
```

and change the gate to:

```python
        if (not rearm
                and getattr(self.controller, "_raster_active", False)
                and getattr(self.controller, "_raster_source", None) == "remote"):
```

`_on_remote_arm_requested` calls `self._start_raster(source="remote")` and is unaffected by the new default. Now add the handler, next to `_start_raster`:

```python
    def _on_rearm_clicked(self) -> None:
        """Swap pending into armed. Deliberately NOT gated on ownership: this
        changes WHICH path is armed, never the cursor, so BLACS's shot count
        cannot desync. The GUI owns the path and needs no permission to
        change it."""
        self._start_raster(source=self._last_raster_source or "local", rearm=True)
```

- [ ] **Step 7: Syntax-check and commit**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m py_compile ui.py raster_controller.py`
Expected: no output, exit 0. Then:

```bash
git commit --only ui.py raster_controller.py -m "feat(ui): draw the armed path, not a frozen preview; add Re-arm

While armed the overlay rendered _raster_preview_pts, which freezes on arm --
so hull vertices kept updating live on top of a path that was no longer being
run, and the operator read coordinates off points nobody was commanding. That
is the 2026-08-07 'motor out of bounds' incident.

The overlay now renders the controller's armed path while armed, the pending
preview refreshes freely on every edit because it can no longer be confused
for the armed one, and an explicit Re-arm swaps pending into armed. Re-arm is
not gated on BLACS ownership: it changes which path is armed, never the
cursor, so it cannot desync BLACS's shot count."
```

---

## Task 5: Say what was dropped

The arm-time reachability filter (already on this branch, `raster_controller.py:1389-1407`) reports via `status_signal` only. BLACS is told nothing.

**Files:**
- Modify: `raster_controller.py:1389-1407` (the filter's status message)
- Modify: `raster_controller.py:556-563` (the `arm_raster` from-scratch reply)

**Interfaces:**
- Produces: `arm_raster` reply gains `extra={"mode": ..., "armed": N, "dropped": M}`.

- [ ] **Step 1: Record the drop count on the controller**

In `raster_controller.py`, in `start_raster`, replace the `if dropped:` status emit with:

```python
            with self._state_lock:
                self._raster_dropped_count = dropped
            if dropped:
                self.status_signal.emit(
                    f"Raster armed: {len(pts)} of {len(pts) + dropped} points; "
                    f"{dropped} dropped, outside motor travel "
                    f"{self.motor_bounds} mm."
                )
```

Initialise `self._raster_dropped_count = 0` in `__init__` alongside `self._raster_total_steps`.

- [ ] **Step 2: Report it on the arm reply**

In `raster_controller.py`, replace the from-scratch arm success reply:

```python
            mode = "continuous" if want_continuous else "step"
            self._outer.status_signal.emit(f"ZMQ: raster armed remotely ({mode}).")
            return encode_reply(
                status="SUCCESS", request_id=request_id,
                extra={"mode": mode},
            )
```

with:

```python
            mode = "continuous" if want_continuous else "step"
            self._outer.status_signal.emit(f"ZMQ: raster armed remotely ({mode}).")
            with self._outer._state_lock:
                armed = len(self._outer._raster_path_pts)
                dropped = getattr(self._outer, "_raster_dropped_count", 0)
            # Unreachable points are dropped, not refused -- a pattern that
            # merely overhangs the frame must not stall a queue. But the drop
            # is never silent: it reaches the BLACS log through this reply.
            return encode_reply(
                status="SUCCESS", request_id=request_id,
                extra={"mode": mode, "armed": armed, "dropped": dropped},
            )
```

- [ ] **Step 3: Run the suite and commit**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py -q`
Expected: PASS, 51 tests.

```bash
git commit --only raster_controller.py -m "feat(raster): report dropped unreachable points to BLACS, not just the GUI log

The arm-time filter drops points that map outside motor travel and said so
only on the GUI status bar. Remote arms now carry armed/dropped counts on the
reply so the drop reaches the BLACS log too. Dropping stays the behaviour --
a pattern that overhangs the frame edge must not stall a queue."
```

---

## Task 6: Dead-zone overlay

~22.7% of the camera frame maps outside 0–12 mm travel under the 2026-07-20 fit, and nothing on screen marks it. This is what stops the pattern being drawn there in the first place.

**Files:**
- Modify: `ui.py` — add `_draw_dead_zone()` and call it from the calibration-ready handler

**Interfaces:**
- Consumes: `controller.calibration`, `controller.motor_bounds`.

- [ ] **Step 1: Add the overlay renderer**

In `ui.py`, next to `_draw_and_enforce_bounds`:

```python
    def _draw_dead_zone(self) -> None:
        """Shade the part of the frame the motors cannot reach. Recomputed on
        every calibration change: the unreachable region is a property of the
        mapping, not of the image, so it moves when the calibration does."""
        for item in getattr(self, "_dead_zone_items", []):
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self._dead_zone_items = []

        cal = getattr(self.controller, "calibration", None)
        bounds = getattr(self.controller, "motor_bounds", None)
        if cal is None or bounds is None:
            return

        w = int(getattr(self, "_frame_w", 500))
        h = int(getattr(self, "_frame_h", 500))
        step = 10  # px; a coarse mask is enough to show the operator the edge
        xs, ys = [], []
        for px in range(0, w, step):
            for py in range(0, h, step):
                if not self.controller._within_bounds(
                        self.controller._target_to_motor_clamped(cal, (px, py)),
                        bounds):
                    xs.append(px)
                    ys.append(py)
        if not xs:
            return
        item = pg.ScatterPlotItem(
            xs, ys, size=step, pxMode=False, pen=None,
            brush=pg.mkBrush(200, 40, 40, 60), symbol="s")
        item.setZValue(-10)   # behind the path overlay and hull vertices
        self.plot_widget.addItem(item)
        self._dead_zone_items = [item]
```

Read how the frame size is obtained elsewhere in `ui.py` (search for `_frame_w`, `config.APP_CONFIG.camera`, or the image item's shape) and use that source rather than the 500 default if one exists.

- [ ] **Step 2: Recompute it when the calibration changes**

Call `self._draw_dead_zone()` at the end of each of these three existing handlers in `ui.py`, immediately after their `self._update_step_mode_ui()` line:
- the calibration-succeeded handler (around `:2193`)
- the calibration-loaded handler (around `:2020`)
- `_reset_calibration_display` (around `:1653`) — this one clears the overlay, since `cal is None` returns early

- [ ] **Step 3: Syntax-check and commit**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m py_compile ui.py`

```bash
git commit --only ui.py -m "feat(ui): shade the region the motors cannot reach

Under the 2026-07-20 fit, 22.7% of the 500x500 frame maps outside 0-12mm
travel -- the left strip plus a wedge along the top -- and nothing on screen
marked it. A convex hull emits its grid x-ascending from the bounding-box
corner, so a hull drawn across the frame leads with the unreachable column.
Recomputed on every calibration change, since the dead zone is a property of
the mapping rather than the image."
```

- [ ] **Step 4: Restart the rastering GUI and confirm visually**

Close and relaunch `python main_rastering.py`. Load the calibration. Expect a translucent red band down the left edge of the image. Draw a hull that crosses into it and confirm arming reports dropped points on the status bar.

---

## Task 7: The Control toggle (parent repo)

**Branch first.** The parent is on `master`, which the operator runs.

```bash
git -C C:/Users/radmo/labscript-suite checkout -b feat/raster-control-toggle
```

**Files:**
- Modify: `userlib/user_devices/RasteringDevice/blacs_tabs.py` — checkbox, layout, slot, save/restore, workerargs, PUB handler

**Interfaces:**
- Produces: worker arg `raster_control` (`"blacs"` | `"local"`, default `"blacs"`); tab attribute `self.raster_control_box`. Task 8 consumes `raster_control`.

- [ ] **Step 1: Add the checkbox to the raster row**

In `blacs_tabs.py`, after the `shots_per_step_box` block:

```python
        # Control: who advances the raster. Orthogonal to Raster Mode -- that
        # gates STEPPING, this gates every remote motor command. "Raster off +
        # Control BLACS" is a real state: remote position feeding with no
        # pattern.
        self.raster_control_box = QtWidgets.QCheckBox("BLACS drives raster")
        self.raster_control_box.setChecked(True)
        self.raster_control_box.setToolTip(
            "Ticked: BLACS advances the raster and may program coordinates.\n"
            "Unticked: the operator drives from the rastering GUI. Shots keep\n"
            "firing; a sequence that programs explicit coordinates will raise.\n"
            "Takes effect on the next shot, not at queue end."
        )
```

and add it to the layout, after `raster_row.addWidget(self.shots_per_step_box)`:

```python
        raster_row.addWidget(self.raster_control_box)
```

and wire it after the two existing `.connect(...)` lines:

```python
        self.raster_control_box.toggled.connect(self.on_raster_control_toggled)
```

- [ ] **Step 2: Add the slot with the correct mode mask**

In `blacs_tabs.py`, after `on_shots_per_step_changed`:

```python
    @define_state(
        MODE_MANUAL | MODE_TRANSITION_TO_BUFFERED | MODE_BUFFERED | MODE_POST_EXP,
        True)
    def on_raster_control_toggled(self, state):
        """Wider mask than the other two raster controls, deliberately.

        queue_state_indefinitely=True does NOT drop a slot fired in a
        disallowed mode -- StateQueue.check_for_next_item only deletes in the
        `not queue_state_indefinitely` branch (tab_base_classes.py:157-160), so
        a MODE_MANUAL-only slot would PARK until the queue ended: the widget
        would read Local while the worker kept stepping. Backwards for a
        control the operator reaches for BECAUSE shots are running.
        MODE_POST_EXP (=32) is the between-queued-shots window.
        """
        yield (
            self.queue_work(
                self.primary_worker, 'update_raster_control',
                raster_control=("blacs" if state else "local"),
            )
        )
```

Add `MODE_TRANSITION_TO_BUFFERED`, `MODE_BUFFERED`, `MODE_POST_EXP` to the existing `define_state` import at the top of the file if they are not already imported.

- [ ] **Step 3: Persist it**

In `get_save_data`, add one key:

```python
        return {
            'raster_mode': self.raster_check_box.isChecked(),
            'shots_per_step': self.shots_per_step_box.value(),
            'raster_control_blacs': self.raster_control_box.isChecked(),
        }
```

In `restore_save_data`, add the `blockSignals` sandwich — mandatory, because `restore_save_data` runs before `initialise_workers`, so `self.primary_worker` is still `None` and an unblocked `toggled` would `queue_work(None, ...)`:

```python
        control_blacs = bool(data.get('raster_control_blacs', True))
        self.raster_control_box.blockSignals(True)
        self.raster_control_box.setChecked(control_blacs)
        self.raster_control_box.blockSignals(False)
```

- [ ] **Step 4: Hand it to the worker**

In `initialise_workers`, add to the workerargs dict after `"shots_per_step"`:

```python
                "raster_control": (
                    "blacs" if self.raster_control_box.isChecked() else "local"),
```

- [ ] **Step 5: Follow the GUI's status without fighting the operator**

In `_on_status_received`, extend the `raster_mode` branch:

```python
        if topic == "raster_mode":
            mode_map = {
                "idle": ("Raster: Idle", "gray"),
                "manual": ("Raster: Manual", "gray"),
                "step": ("Raster: Step", "green"),
                "continuous": ("Raster: Continuous", "yellow"),
            }
            text, color = mode_map.get(value, (f"Raster: {value}", "gray"))
            self.raster_mode_indicator.update_status(text, color)
            # Mirror ownership the operator may have changed at the GUI. The
            # blockSignals sandwich is what keeps a PUB repaint from firing the
            # operator's toggled slot -- same guard restore_save_data uses.
            if value in ("manual", "step", "continuous"):
                self.raster_control_box.blockSignals(True)
                self.raster_control_box.setChecked(value != "manual")
                self.raster_control_box.blockSignals(False)
```

- [ ] **Step 6: Syntax-check**

Run:
```
& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n labscript --no-capture-output python -m py_compile C:/Users/radmo/labscript-suite/userlib/user_devices/RasteringDevice/blacs_tabs.py
```
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git -C C:/Users/radmo/labscript-suite commit --only userlib/user_devices/RasteringDevice/blacs_tabs.py -m "feat(RasteringDevice): Control toggle, independent of Raster Mode

One checkbox was doing two jobs. Raster Mode gates STEPPING; the new Control
toggle gates every remote motor command, so 'Raster off + Control BLACS' is a
real state -- remote position feeding with no pattern.

The mask is wider than the other raster controls on purpose:
queue_state_indefinitely=True does not drop a slot fired in a disallowed mode
(tab_base_classes.py:157-160), it PARKS it, so a MODE_MANUAL-only slot would
read Local in the widget while the worker kept stepping to queue end.

The checkbox both shows and sets; a PUB-driven repaint is wrapped in the same
blockSignals sandwich restore_save_data already uses, so incoming status can
never fire the operator's toggled slot."
```

---

## Task 8: Gate the worker on Control (parent repo)

**Files:**
- Modify: `userlib/user_devices/RasteringDevice/blacs_workers.py` — `_init_raster_state`, new `update_raster_control`, `_sync_raster_mode_to_gui`, `_advance_raster`, `transition_to_buffered`

**Interfaces:**
- Consumes: worker arg `raster_control` from Task 7; the GUI replies from Tasks 1, 3, 5.

- [ ] **Step 1: Normalize the new worker arg**

In `blacs_workers.py`, in `_init_raster_state`, next to the existing normalizations:

```python
        # "blacs" | "local". Workerargs are set as instance attributes before
        # init() runs, so normalize rather than clobber.
        self.raster_control = str(getattr(self, "raster_control", "blacs"))
```

- [ ] **Step 2: Add the settings-change hook**

In `blacs_workers.py`, next to `update_raster_mode`:

```python
    def update_raster_control(self, raster_control):
        """Control changed at the tab. Releasing to the operator sends
        disarm_raster, which the GUI now treats as 'release ownership,
        keep the path' rather than 'destroy the raster'."""
        self.raster_control = str(raster_control)
        if self.raster_control == "local":
            self._raster_armed = False
            try:
                response = self.remote_comms.program_value("disarm_raster", 1)
                self._check_response(response, "raster_release")
            except Exception as e:
                # Never raise from a settings change: the GUI may be closed.
                self.logger.warning(f"Could not release raster to GUI: {e}")
        else:
            self._sync_raster_mode_to_gui()
```

- [ ] **Step 3: Gate all three arm senders**

`arm_raster`'s already-armed branch sets `_raster_source = "remote"` on the GUI (`raster_controller.py:484`), so any un-gated arm sender seizes control back. There are three.

In `_sync_raster_mode_to_gui`, change the arm condition:

```python
        if self.raster_mode:
```
to
```python
        if self.raster_mode and self.raster_control == "blacs":
```

In `_advance_raster`, change the lazy-arm condition:

```python
                if not self._raster_armed:
```
to
```python
                if not self._raster_armed and self.raster_control == "blacs":
```

In `connect_to_remote`, the re-sync at `:74` is already routed through `_sync_raster_mode_to_gui`, so it inherits the gate — confirm by reading it and add a comment noting the dependency.

- [ ] **Step 4: Ask every shot under local control**

In `_advance_raster`, replace:

```python
        if self._shots_since_step == 0:
```

with:

```python
        # Under local control the group counter has no job -- BLACS is not
        # advancing anything, it is only asking where the laser is. Skipping
        # shots 2..N of a group would stamp them with a point the operator has
        # since stepped away from. Keep mutating the counter below so the group
        # phase survives a flip back to BLACS mid-queue.
        if self._shots_since_step == 0 or self.raster_control == "local":
```

- [ ] **Step 5: Gate the buffered coordinate write on Control**

In `transition_to_buffered`, before the `for connection, value in writes:` loop:

```python
                if self.raster_control == "local":
                    raise Exception(
                        "Sequence programs explicit raster coordinates, but "
                        "Control is set to Local.\n"
                        "The operator is hand-driving the raster; a remote "
                        "position write would fight them.\n"
                        "Tick 'BLACS drives raster' in this tab (or hand "
                        "control back at the rastering GUI) and resume."
                    )
```

Note this is gated on `raster_control`, NOT `raster_mode` — "Raster off + Control BLACS" must still allow remote position feeding.

- [ ] **Step 6: Syntax-check**

Run:
```
& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n labscript --no-capture-output python -m py_compile C:/Users/radmo/labscript-suite/userlib/user_devices/RasteringDevice/blacs_workers.py
```
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git -C C:/Users/radmo/labscript-suite commit --only userlib/user_devices/RasteringDevice/blacs_workers.py -m "feat(RasteringDevice): honour Control in every path that touches the motors

Gating move_to_next alone would not have worked: arm_raster's already-armed
branch sets ownership to remote on the GUI (raster_controller.py:484), and
three separate senders reach it -- the eager sync on tick, the lazy arm inside
transition_to_buffered, and the reconnect re-sync. All three are gated now.

Under local control the worker asks every shot rather than once per group, so
a shot cannot be stamped with a point the operator has since stepped away from.

The buffered coordinate write raises under Control=Local rather than fighting
the operator. It is gated on Control, not Raster Mode, so 'Raster off +
Control BLACS' still allows remote position feeding with no pattern."
```

---

## Deliberately not in this plan

- **`raster_control: "blacs"|"local"` stamped into the shot h5** in `post_experiment`. The
  spec marks it optional — `raster_point_meta` already carries `target_xy`, `point_index`
  and the calibration, which is enough to reconstruct where any shot fired. Add it only if
  analysis turns out to need to distinguish who drove.

## Deploy and verify

- [ ] **Restart the rastering GUI first, then BLACS.** BLACS must not send a Control-gated flow to an old GUI.
- [ ] **Flip Control to Local mid-queue** → the *next* shot honours it. If it only takes effect at queue end, the Task 7 mode mask is wrong.
- [ ] **Untick Raster Mode** → the armed path survives and the indicator reads `Raster: Manual`.

Everything else — GUI Step under local control, the sequence-coords raise, the dead-zone shading — announces itself on first use.
