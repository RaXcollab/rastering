# Raster Arming Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every confirmed defect from the 2026-08-07 post-implementation review of the raster arming overhaul: a red test suite committed to the GUI branch, three ownership-consistency holes, an invisible dead-zone overlay, a display path that can still lie while armed, and a BLACS tab mirror that updates its widget but not its worker.

**Architecture:** No design change — this closes gaps between the approved spec (`docs/superpowers/specs/2026-08-07-raster-arming-design.md`) and the committed code, plus three adjudicated corrections to the spec itself. One deliberate scope addition: a `raster_owner` PUB value (`local`/`remote`/`none`) published on the existing PUB socket, because the existing `raster_mode` topic conflates run-mode with ownership and cannot drive the tab's Control checkbox correctly (a locally-owned continuous raster publishes `continuous`, not `manual`). It is a broadcast on an existing socket, not a new request/reply exchange.

**Tech Stack:** Python 3.11, PyQt5, pyqtgraph, pyzmq v2 protocol, pytest. GUI = conda env `rastering`; BLACS = conda env `labscript`.

## Global Constraints

- **TWO camera-safe test files, not one** (this corrects the repo CLAUDE.md, which Task 1 also fixes): `tests/test_raster_pathmodel.py` AND `tests/test_zmq_v2_protocol.py`. Neither imports `ui.py`. Run both after every GUI-repo task:
  `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py -q`
- **NEVER run the whole `tests/` directory.** `test_command_queue.py` and `test_raster_goto_handlers.py` import `ui.py` → open the uEye camera → HANG while the operator's GUI runs.
- **NEVER import `ui.py`.** Syntax-check with `python -m py_compile ui.py`.
- **Never stage, commit, or restore `calibration_data.json`.** Commit with `git commit --only <explicit paths>`.
- GUI tasks (1–4) run in `C:/Users/radmo/labscript-suite/GUIs/rastering-hull-reach` (branch `fix/raster-reachability`).
- BLACS tasks (5–6) run in `C:/Users/radmo/labscript-suite/.claude/worktrees/raster-control` (branch `feat/raster-control-toggle`). Prefix every git command with `git -C <that path>`. NEVER touch `C:/Users/radmo/labscript-suite/userlib/` — that is the operator's live master checkout.
- BLACS side has no runnable test suite; verify with `conda run -n labscript python -m py_compile <file>`.
- If a "replace this" block does not byte-match the file, STOP that step and report the actual text in your deviations — only make the semantically identical edit if it is unambiguous.
- Baseline at plan time: GUI suite = pathmodel 51 passed; zmq_v2 **3 failed**, 46 passed. The three failures are known and fixed by Tasks 1–2.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `tests/test_zmq_v2_protocol.py` | fixture ownership param; re-pin finished/disarm/not-active contracts | 1, 2 |
| `CLAUDE.md` (GUI repo) | correct the camera-safe test list | 1 |
| `raster_controller.py` | `_finish_raster` ownership, in-place broadening + payload, TOCTOU re-read, honest never-stepped meta, `raster_owner` PUB, `give_remote_control` | 2, 3 |
| `tests/test_raster_pathmodel.py` | finish-preserves-ownership; never-stepped meta update | 2 |
| `ui.py` | pending overlay split, param-change wipe fix, Re-arm ownership, dead-zone z/reshape, tooltips, hand-back button | 3 |
| `docs/superpowers/specs/2026-08-07-raster-arming-design.md` | 3 adjudicated corrections | 4 |
| `RasteringDevice/blacs_workers.py` (worktree) | release-on-connect, arm-reply logging, META_KEYS, buffered-gate precision | 5 |
| `RasteringDevice/blacs_tabs.py` (worktree) | `raster_owner` subscription; mirror that syncs the worker | 6 |

---

## Task 1: Make the zmq_v2 suite green — and make it pin the NEW contracts

The branch was committed with `tests/test_zmq_v2_protocol.py` red (3 failures) because the plan's constraints — copied from a factually wrong CLAUDE.md line — said the file wasn't safe to run. Fix the fixture, re-pin the three tests to the new semantics, and correct CLAUDE.md so this cannot recur.

**Files:**
- Modify: `tests/test_zmq_v2_protocol.py:61-83` (fixture), `:435-447` (iter-end), `:475-482` (not-active), `:535-548` (disarm)
- Modify: `CLAUDE.md` (Python Environment section, Tests line)

**Interfaces:**
- Produces: `_make_outer(..., raster_source="__default__")` — explicit ownership control for every test. Task 2's new tests consume it.

- [ ] **Step 1: Parameterize ownership in the fixture and give it the new controller attribute**

In `tests/test_zmq_v2_protocol.py`, change the `_make_outer` signature:

```python
def _make_outer(*, target_xy=None, motor_xy=None,
                raster_active=False, raster_has_path=False,
                raster_continuous=False, move_x_ok=True, move_y_ok=True,
                move_pair_ok=True,
                step_returns=None, raster_step_calls=None,
                raster_source="__default__"):
```

and replace the ownership line at `:77`:

```python
    outer._raster_source = "local" if raster_active else None
```

with:

```python
    # Ownership is now a persistent flag (2026-08-07 overhaul), so tests must
    # be able to say who holds it. Default keeps the historical shape: armed
    # fixtures were built by a local arm.
    if raster_source == "__default__":
        raster_source = "local" if raster_active else None
    outer._raster_source = raster_source
    # Real int, set in SystemController.__init__ since the arm-time filter
    # started reporting drop counts. A MagicMock auto-child here is not JSON
    # serializable and turns every from-scratch arm reply into an ERROR.
    outer._raster_dropped_count = 0
```

Confirm `make_v2_pair` forwards `**kwargs` to `_make_outer` (read the fixture); if it enumerates kwargs explicitly, add `raster_source` there too.

- [ ] **Step 2: Run the previously failing arm test to verify the fixture fix**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_zmq_v2_protocol.py::test_v2_arm_raster_remote_provider_success_then_step -q`
Expected: PASS (it failed only because `_raster_dropped_count` was a MagicMock in the arm reply).

- [ ] **Step 3: Re-pin the iterator-end contract under BLACS drive**

The iter-end reply is only `finished:True` when BLACS holds the raster; under local hold the same `res is None` is an acknowledge. The test must say which case it pins. In `test_v2_program_value_move_to_next_iter_end_returns_finished_extra`, change the fixture call to:

```python
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        raster_source="remote",           # exhausted while BLACS drives
        step_returns=lambda **kw: None,   # iterator end
    )
```

and extend the docstring's last line with: `Under local hold the same None is an acknowledge, pinned separately.`

- [ ] **Step 4: Pin not-active-under-BLACS as the error case**

`test_v2_program_value_move_to_next_not_active_rejected` currently builds `raster_active=False` (source `None`). Task 2 makes source-`None` fire in place, so this test must pin the *remote* case explicitly. Change its fixture call to:

```python
    outer, client_t, v2_server = make_v2_pair(raster_active=False,
                                              raster_source="remote")
```

and rename it `test_v2_move_to_next_not_active_under_blacs_rejected` with docstring: `"""Nothing armed while BLACS drives is an error -- it is what triggers BLACS's re-arm self-heal. The local/unset cases fire in place instead (pinned separately)."""`

- [ ] **Step 5: Rewrite the disarm test for release semantics**

Replace `test_v2_disarm_raster_while_active_step_mode_stops_it` (`:535-548`) entirely with:

```python
def test_v2_disarm_raster_while_active_releases_to_local(make_v2_pair):
    """disarm_raster releases ownership and PRESERVES the armed path.
    BLACS unticking Raster Mode means 'I stop driving', not 'destroy the
    operator's pattern' -- only the GUI Stop button destroys (2026-08-07)."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_has_path=True, raster_continuous=False,
        raster_source="remote")
    outer._remote_shots_per_step = 4
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 52, "action": "PROGRAM_VALUE",
        "connection": "disarm_raster", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["disarmed"] is True
    outer.stop_raster.assert_not_called()
    assert outer._raster_source == "local"
    assert outer._raster_path_pts, "armed path must survive a release"
    outer.raster_source_signal.emit.assert_called_once_with("local")
    # Shots-per-step is meaningless once released -> back to "--" in the GUI.
    assert outer._remote_shots_per_step is None
    outer.raster_shots_per_step_signal.emit.assert_called_once_with(None)
```

- [ ] **Step 6: Correct CLAUDE.md**

In the GUI repo's `CLAUDE.md`, replace the Tests line:

```
- **Tests:** only `pytest tests/test_raster_pathmodel.py` is camera-safe (pure path/controller logic; runs in CI). `test_command_queue.py` and `test_raster_goto_handlers.py` import `ui.py` → open the uEye camera → **HANG when the GUI/camera is busy** — never run them (or the whole `tests/` dir) while the rastering GUI runs. Use `python -m py_compile` for syntax. Tests are standalone-runnable.
```

with:

```
- **Tests:** `pytest tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py` are BOTH camera-safe (pure controller/protocol logic, no `ui.py` import; run BOTH before any commit). `test_command_queue.py` and `test_raster_goto_handlers.py` import `ui.py` → open the uEye camera → **HANG when the GUI/camera is busy** — never run them (or the whole `tests/` dir) while the rastering GUI runs. Use `python -m py_compile` for syntax. Tests are standalone-runnable.
```

- [ ] **Step 7: Run both suites**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n rastering --no-capture-output python -m pytest tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py -q`
Expected: **100 passed** (51 + 49), 0 failed.

- [ ] **Step 8: Commit**

```bash
git commit --only tests/test_zmq_v2_protocol.py CLAUDE.md -m "test(zmq): re-pin move_to_next/disarm contracts for ownership semantics; fix camera-safe list

The overhaul branch was committed with this file red: the plan's constraints
said only test_raster_pathmodel.py was camera-safe, copied from a CLAUDE.md
line that is factually wrong -- this file imports no ui.py and runs in ~1.5s.
CLAUDE.md now names both camera-safe files.

Fixture gains an explicit raster_source param (ownership is persistent now,
so every protocol test must say who holds the raster) and a real
_raster_dropped_count (the MagicMock auto-child broke JSON encoding of the
from-scratch arm reply). Iter-end and not-active tests pin the BLACS-drive
cases explicitly; the disarm test now pins release-preserves-path."
```

---

## Task 2: Ownership honesty — close the three controller holes

Three defects in `raster_controller.py`: `_finish_raster` still writes ownership (a machine event), the fire-in-place gate misses the fresh-GUI `None` case and carries no position, and a handler/step TOCTOU can turn an operator click into a spurious `finished` (which makes BLACS restart the pattern at point 1). Plus: the never-stepped meta fabricates path-point-0 coordinates the laser never visited.

**Files:**
- Modify: `raster_controller.py:571-612` (move_to_next handler), `:1587-1600` (raster_point_meta), `:2352-2365` (_finish_raster), PUB block (~`:2455-2470`)
- Test: `tests/test_raster_pathmodel.py`, `tests/test_zmq_v2_protocol.py`

**Interfaces:**
- Consumes: `raster_source=` fixture param from Task 1.
- Produces: PUB topic `raster_owner` with values `"local"`/`"remote"`/`"none"` at ~1 Hz (Task 6 consumes); in-place reply shape `{"in_place": True, "frame": ..., "target_xy": [...]}` (Task 5 whitelists `in_place`); `raster_point_meta` never-stepped shape `point_index=-1` + real cached position.

- [ ] **Step 1: Write the failing tests**

In `tests/test_raster_pathmodel.py` (extend `_step_self` minimally if `_finish_raster` touches a signal it lacks — it emits `raster_state_signal`, `raster_source_signal`, `raster_finished_signal`, `status_signal`, `selection_changed_signal` and calls `_flush_raster_log`):

```python
def test_finish_raster_preserves_ownership():
    """Path exhaustion is a machine event. Ownership is a human decision and
    must survive _finish_raster -- otherwise fire-in-place dies in exactly
    the case it was built for (operator driving, pattern ran out)."""
    sc = _step_self([(1.0, 2.0)], active=True)
    sc._raster_source = "local"
    sc.raster_finished_signal = mock.Mock()
    SystemController._finish_raster(sc)
    assert sc._raster_active is False
    assert sc._raster_source == "local"
    sc.raster_source_signal.emit.assert_not_called()
```

In `tests/test_zmq_v2_protocol.py`:

```python
def test_v2_move_to_next_not_active_unset_owner_fires_in_place(make_v2_pair):
    """Fresh GUI (_raster_source None, nothing armed): a move_to_next must
    fire in place, not raise -- reaching here with nothing armed means BLACS
    is NOT in control of arming (Control=Local), because under Control=BLACS
    the worker always arms before stepping."""
    outer, client_t, v2_server = make_v2_pair(raster_active=False)
    outer.calibration = None
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 60, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["in_place"] is True


def test_v2_move_to_next_not_active_local_carries_position(make_v2_pair):
    """Fire-in-place must record the real site: the reply carries the cached
    target position + frame so the shot h5 is not empty for these shots."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=False, raster_source="local", target_xy=(3.0, 4.0))
    outer.calibration = None
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 61, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["in_place"] is True
    assert reply["target_xy"] == [3.0, 4.0]
    assert reply["frame"] == "motor"


def test_v2_move_to_next_armed_local_acks_current_point(make_v2_pair):
    """Armed + operator holds control: res None is an ACK with current-point
    meta, never finished:True -- a spurious finished makes BLACS clear its
    armed flag, re-arm from scratch, and restart the pattern at point 1."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        raster_source="local", step_returns=lambda **kw: None)
    outer.raster_point_meta = mock.Mock(return_value={
        "point_index": 3, "path_len": 9, "frame": "pixel",
        "target_xy": [1.0, 2.0]})
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 62, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert "finished" not in reply
    assert reply["point_index"] == 3
```

If `_make_outer` already stubs `raster_point_meta`, the explicit override above still wins — keep it for determinism.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `...pytest tests/test_zmq_v2_protocol.py -q -k "in_place or acks_current"` → the unset-owner test FAILS (status ERROR); the position test FAILS (`KeyError: 'target_xy'`). Run `...pytest tests/test_raster_pathmodel.py -q -k finish_raster_preserves` → FAILS (`assert None == 'local'`).

- [ ] **Step 3: `_finish_raster` stops writing ownership**

In `raster_controller.py`, in `_finish_raster`, delete the two lines `self._raster_source = None` (`:2359`) and `self.raster_source_signal.emit(None)` (`:2361`), and add above the state writes:

```python
            # _raster_source is NOT cleared: path exhaustion is a machine
            # event, and ownership changes only when a human changes it --
            # same contract as stop_raster. Clearing here silently dropped
            # the operator's hold the moment their pattern ran out.
```

- [ ] **Step 4: Broaden and enrich fire-in-place; kill the TOCTOU**

In the `move_to_next` handler, replace `:571-612`:

```python
        if connection == "move_to_next":
            with self._outer._state_lock:
                active = self._outer._raster_active
                continuous = self._outer._raster_continuous
                held_by_operator = self._outer._raster_source == "local"
            if not active:
                if held_by_operator:
```

...through the end of the `res is None` block (`:612`), with:

```python
        if connection == "move_to_next":
            with self._outer._state_lock:
                active = self._outer._raster_active
                continuous = self._outer._raster_continuous
                source = self._outer._raster_source
            held_by_operator = source == "local"
            if not active:
                if source != "remote":
                    # Fire in place -- for "local" AND for unset (fresh GUI).
                    # Reaching here with nothing armed means BLACS is not in
                    # control of arming: under Control=BLACS the worker arms
                    # before every step, so an unset owner can only mean the
                    # operator side. Per-shot stepping (42c815f) made this a
                    # hard failure; before it, every shot fired in place.
                    # The reply carries the REAL cached position so the shot
                    # h5 records where the laser actually sat.
                    with self._outer._state_lock:
                        txy = self._outer._last_target_xy
                        cal = self._outer.calibration
                    extra = {"in_place": True,
                             "frame": "pixel" if cal is not None else "motor"}
                    if txy is not None:
                        extra["target_xy"] = [float(txy[0]), float(txy[1])]
                    return encode_reply(
                        status="SUCCESS", request_id=request_id, extra=extra,
                    )
                return self._err(
                    request_id=request_id, code="raster_not_active",
                    message="raster not active",
                )
            if continuous:
                return self._err(
                    request_id=request_id, code="raster_in_continuous_mode",
                    message="raster in continuous mode",
                )
            res = self._outer.raster_step(
                source="zmq", wait=True, timeout_s=timeout_sec)
            if res is None:
                # Re-read ownership: a take_local_control() landing between
                # the first read and the step would otherwise surface as
                # finished:True, which BLACS answers by re-arming from
                # scratch and restarting the pattern at point 1 -- a whole
                # raster restarted by one operator click at the wrong ms.
                with self._outer._state_lock:
                    held_now = self._outer._raster_source == "local"
                if held_by_operator or held_now:
                    # Armed, operator driving: acknowledge with the CURRENT
                    # point so the shot h5 records where the laser actually
                    # is, without moving the cursor BLACS is not driving.
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

(The `res.ok` / `raster_step_failed` tail below stays untouched.)

- [ ] **Step 5: Honest never-stepped meta**

In `raster_point_meta`, add `txy = self._last_target_xy` to the locked read (`:1578-1582`), then replace the clamp block (`:1592-1597`):

```python
        if i < 0:
            # Armed but never stepped: report the point the raster WILL fire
            # at (point 0) rather than -1 with no coordinates, which would
            # land a bogus record in the shot h5.
            meta["point_index"] = 0
            pt = self._raster_path_pts[0] if total else None
```

with:

```python
        if i < 0:
            # Armed but never stepped: point_index -1 is the honest value
            # ("not on a path point yet"). The coordinate reported is the
            # laser's ACTUAL cached position -- NOT path point 0, which
            # nothing has moved to. Point-0 coords here would be plausible
            # and wrong in the shot h5, the worst kind of record.
            pt = txy
```

Then find the pathmodel test the overhaul added for this clamp (`grep -n "point_index" tests/test_raster_pathmodel.py`) and update its asserts to the new contract: `meta["point_index"] == -1` and `meta["target_xy"]` equal to the fixture's `_last_target_xy` (set one on the stub if absent). Record the exact test name you changed in your deviations.

- [ ] **Step 6: Publish `raster_owner`**

In the PUB status block (the `if pub_counter % 4 == 0:` body, after the `raster_mode` publishes around `:2470`), add:

```python
                    # Ownership on its own value: raster_mode conflates
                    # run-mode with ownership (a locally-owned CONTINUOUS
                    # raster publishes "continuous", not "manual"), so the
                    # BLACS tab's Control checkbox cannot be driven from it.
                    publish("raster_owner", source or "none")
```

(`source` is already read under the lock in that block since the `manual` change; confirm, else add it to the locked read.)

- [ ] **Step 7: Run both suites**

Run: `...pytest tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py -q`
Expected: **104 passed** (52 + 52), 0 failed. (51+1 pathmodel — the never-stepped test is updated, not added; 49+3 zmq_v2.)

- [ ] **Step 8: Commit**

```bash
git commit --only raster_controller.py tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py -m "fix(raster): ownership survives finish; fire-in-place is honest and race-free

_finish_raster no longer clears _raster_source -- path exhaustion is a
machine event, and clearing dropped the operator's hold the moment their
pattern ran out, killing fire-in-place in exactly the case it exists for.

Fire-in-place now also covers an unset owner (fresh GUI): reaching
move_to_next with nothing armed means BLACS is not in control of arming,
since under Control=BLACS the worker arms before every step. The reply
carries the cached position + frame so those shots record their real site.

res-None replies re-read ownership before answering: a take_local_control
landing mid-step no longer surfaces as finished:True, which BLACS answers
by re-arming from scratch and restarting the pattern at point 1.

raster_point_meta on a never-stepped raster reports point_index -1 with the
laser's ACTUAL cached position instead of fabricating path point 0's coords.

New raster_owner PUB value (local/remote/none): raster_mode conflates
run-mode with ownership, so the BLACS tab's Control mirror cannot be driven
from it."
```

---

## Task 3: The display stops lying — completely

Four `ui.py` defects: a pending-render failure wipes the armed dots permanently; direction lines trace the pending path over armed dots; Re-arm inherits a sticky `"remote"` and hands the operator's pattern to BLACS; the dead-zone overlay paints behind the opaque camera image. Plus stale tooltips that describe the deleted last-stepper-wins world, and no GUI-side route to hand an armed raster back to BLACS.

**Files:**
- Modify: `ui.py:46-63` (tooltips), `~:830-835` (stale comment), `:869-882` (button enable), `:927-945` (`_arm_for_remote`), `:1458-1494` (`_draw_dead_zone`), `:265-271` (frame-shape hook), `:1598-1615` (`_on_raster_param_changed`), `:1638-1643` (`_on_rearm_clicked`), `_render_preview` (`:1531-1596`), `_refresh_raster_scatter` (`:2177-2199`)
- Modify: `raster_controller.py` (add `give_remote_control`, next to `take_local_control`)

**Interfaces:**
- Consumes: `armed_path_points()` (exists), `take_local_control()` (exists).
- Produces: `SystemController.give_remote_control() -> bool` (mirror of `take_local_control`); `ui` attributes `pending_scatter` (pg.ScatterPlotItem, grey open circles) and `_clear_pending_overlay()`.

- [ ] **Step 1: Add `give_remote_control` to the controller**

In `raster_controller.py`, directly below `take_local_control`:

```python
    def give_remote_control(self) -> bool:
        """Operator hands an armed raster to BLACS from the GUI side -- the
        mirror of take_local_control, so the Control axis is settable from
        either screen. Returns False and touches nothing when no raster is
        active (there is nothing to hand over)."""
        with self._state_lock:
            if not self._raster_active:
                return False
            self._raster_source = "remote"
        self.raster_source_signal.emit("remote")
        return True
```

- [ ] **Step 2: Pending gets its own scatter; direction lines only while idle**

In `ui.py`, in `_render_preview`, the drawing tail currently caches `self._raster_preview_pts = [...]`, updates `goto_index_spin`, calls the scatter refresh and draws direction lines from `pts`. Rework that tail so:

```python
        # Cache the full preview so the Display-Options filter can re-render
        # the overlay on toggle without regenerating the iterator.
        self._raster_preview_pts = [(float(p[0]), float(p[1])) for p in pts]
        armed = bool(getattr(self, "_raster_active_ui", False))
        if armed:
            # Pending-while-armed: its OWN grey overlay, never the armed one.
            # Direction lines are deliberately skipped -- grey lines tracing
            # the pending pattern over the armed dots is exactly the
            # read-off-the-wrong-path failure this split exists to prevent.
            self._ensure_pending_scatter()
            self.pending_scatter.setData(
                [p[0] for p in self._raster_preview_pts],
                [p[1] for p in self._raster_preview_pts])
        else:
            <existing scatter refresh + direction-line drawing, unchanged>
        <existing goto_index_spin block, unchanged>
```

The exact existing tail differs from this sketch — keep every existing statement, only fence the scatter refresh and the direction-line `for` loop behind `if not armed:` and add the pending branch. Add the two helpers next to `_clear_raster_overlay`:

```python
    def _ensure_pending_scatter(self) -> None:
        """Lazy pending overlay: grey open circles, visually distinct from the
        armed path's filled dots so the two can NEVER be confused."""
        if getattr(self, "pending_scatter", None) is None:
            self.pending_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen("#999999", width=1), brush=None,
                symbol="o", size=6)
            self.plot_widget.addItem(self.pending_scatter)

    def _clear_pending_overlay(self) -> None:
        """Clear ONLY the pending (grey) overlay + its cache. Never touches
        raster_scatter: while armed that shows the running path, and a
        pending-side clear must not be able to blank it."""
        self._raster_preview_pts = []
        if getattr(self, "pending_scatter", None) is not None:
            self.pending_scatter.clear()
```

- [ ] **Step 3: Param changes can no longer wipe the armed display**

Replace `_on_raster_param_changed`'s body from `if not self._raster_preview_pts:` down (`:1611-1615`):

```python
        if not self._raster_preview_pts:
            return
        self._clear_raster_overlay()
        self._render_preview(quiet=True)
        self._update_armed_pending_status()
```

with:

```python
        if not self._raster_preview_pts:
            return
        armed = bool(getattr(self, "_raster_active_ui", False))
        if armed:
            # Pending-only clear: _clear_raster_overlay would blank
            # raster_scatter, and every early return in _render_preview
            # (spec raise, hull<3, 0 points) would then leave the ARMED
            # path invisible for the rest of the run.
            self._clear_pending_overlay()
        else:
            self._clear_raster_overlay()
        self._render_preview(quiet=True)
        if armed:
            # Unconditional: the armed dots are re-asserted even when the
            # pending render bailed early.
            self._refresh_raster_scatter()
        self._update_armed_pending_status()
```

- [ ] **Step 4: Arm/stop transitions clear the pending overlay**

Find the slot connected to `raster_state_signal` (search `raster_state_signal.connect` in the controller-signal wiring; the slot toggles `_raster_active_ui`). Append to it:

```python
        # Arming consumes the pending pattern (it IS the armed one now);
        # stopping returns the plot to a single normal preview. Either way
        # the grey overlay is stale the moment the state flips.
        self._clear_pending_overlay()
        self._refresh_raster_scatter()
```

Also append the same two lines at the end of `_start_raster` (after `self._log(f"Raster started: {spec.kind}")`) so a Re-arm's swap is immediate even before the queued signal lands.

- [ ] **Step 5: Re-arm keeps live ownership, never a stale one**

Replace `_on_rearm_clicked` (`:1638-1643`):

```python
    def _on_rearm_clicked(self) -> None:
        """Swap pending into armed. Deliberately NOT gated on ownership: this
        changes WHICH path is armed, never the cursor, so BLACS's shot count
        cannot desync. The GUI owns the path and needs no permission to
        change it."""
        self._start_raster(source=self._last_raster_source or "local", rearm=True)
```

with:

```python
    def _on_rearm_clicked(self) -> None:
        """Swap pending into armed. Deliberately NOT gated on ownership: this
        changes WHICH path is armed, never the cursor, so BLACS's shot count
        cannot desync. Ownership: carry the LIVE flag through when a raster
        is active (BLACS keeps driving the swapped path); arm as local when
        idle. Never _last_raster_source -- that is sticky across Stop (stop
        preserves ownership since 2026-08-07), so a stale 'remote' would hand
        a freshly drawn pattern to BLACS and lock the operator out of Step."""
        if getattr(self.controller, "_raster_active", False):
            src = getattr(self.controller, "_raster_source", None) or "local"
        else:
            src = "local"
        self._start_raster(source=src, rearm=True)
```

- [ ] **Step 6: Dead zone becomes visible and tracks the frame**

In `_draw_dead_zone`, replace:

```python
        item.setZValue(-10)   # behind the path overlay and hull vertices
        self.plot_widget.addItem(item)
```

with:

```python
        # Stacking: the camera ImageItem and every overlay default to z=0,
        # ordered by insertion -- an item added later at z<=-? is a trap:
        # ViewBox only re-bumps z below its own -100. Pin the image UNDER
        # everything once, and sit the shading between image and overlays.
        if hasattr(self, "img_item"):
            self.img_item.setZValue(-1)
        item.setZValue(-0.5)  # above the image, below all z=0 overlays
        self.plot_widget.addItem(item)
```

(`img_item` is the `pg.ImageItem` created in the image setup around `:192`; if the attribute has a different name there, use that name and note it.) Then at the frame-ingest site (`:265-271`, where `self._last_frame_shape = (h, w)` is assigned), make the assignment shape-aware:

```python
            new_shape = (h, w)
            shape_changed = new_shape != getattr(self, "_last_frame_shape", None)
            self._last_frame_shape = new_shape
            if shape_changed and getattr(self, "_dead_zone_items", None):
                # AOI / camera-settings changes move the frame under the
                # shading; recompute so unreachable columns are never
                # unmarked (or phantom-marked) after a reshape.
                self._draw_dead_zone()
```

- [ ] **Step 7: Tooltips and comments stop describing the deleted bug**

Replace `_TAKE_BACK_TIP` (`:46-51`):

```python
_TAKE_BACK_TIP = (
    "Take the raster back: Control returns to Local and holds -- BLACS steps\n"
    "are acknowledged without advancing until you hand control back (here or\n"
    "in the BLACS tab). Stop tears the path down for real.")
```

Replace `_GOTO_TAKEOVER_TIP` (`:60-63`):

```python
_GOTO_TAKEOVER_TIP = (
    "Move to the selected raster point.\n"
    "BLACS owns this raster -- moving takes local control, and it HOLDS: "
    "BLACS cannot reclaim it by stepping. Hand it back with 'Give to BLACS' "
    "or the tab's Control toggle.")
```

Find the stale comment inside `_update_step_mode_ui` (around `:830-835`) reading `... (raster_step hands ownership to whoever stepped last). "Return to local control" and Stop are the ways back.` and rewrite the parenthetical to: `(ownership is a persistent flag only human actions change -- a BLACS step can no longer seize it back)`.

- [ ] **Step 8: A GUI-side route back to BLACS**

In `_update_step_mode_ui`, replace the else-branch of the dual-face button (`:878-882`):

```python
            else:
                self.raster_remote_arm_button.setText("Arm for remote stepping")
                self.raster_remote_arm_button.setEnabled(calibrated and not active)
                self.raster_remote_arm_button.setToolTip(
                    _ARM_TIP if calibrated else _cal_hint)
```

with:

```python
            elif active:
                # Armed and locally held: third face -- hand it back. Without
                # this the tab checkbox is the ONLY route back to BLACS, and
                # spec section 2 promises the axis is settable from either
                # screen.
                self.raster_remote_arm_button.setText("Give to BLACS")
                self.raster_remote_arm_button.setEnabled(True)
                self.raster_remote_arm_button.setToolTip(
                    "Hand this armed raster to BLACS: its next move_to_next "
                    "advances from the current point.")
            else:
                self.raster_remote_arm_button.setText("Arm for remote stepping")
                self.raster_remote_arm_button.setEnabled(calibrated)
                self.raster_remote_arm_button.setToolTip(
                    _ARM_TIP if calibrated else _cal_hint)
```

and in `_arm_for_remote`, replace the active-raster branch (`:938-943`):

```python
        if getattr(self, "_raster_active_ui", False):
            if self._last_raster_source == "remote":
                self.controller.take_local_control()
                return
            self._log("Raster is already active; press Stop before re-arming for remote stepping.")
            return
```

with:

```python
        if getattr(self, "_raster_active_ui", False):
            if self._last_raster_source == "remote":
                self.controller.take_local_control()
            else:
                # Third face: hand the armed raster to BLACS in place.
                self.controller.give_remote_control()
            return
```

- [ ] **Step 9: Syntax-check, run suites, commit**

Run: `...python -m py_compile ui.py raster_controller.py` then both test suites.
Expected: exit 0; **104 passed**.

```bash
git commit --only ui.py raster_controller.py -m "fix(ui): pending overlay is its own layer; Re-arm keeps live ownership; dead zone visible

Pending renders into a dedicated grey open-circle scatter while armed, and
direction lines draw only while idle -- grey lines tracing the pending
pattern over armed dots was the read-off-the-wrong-path failure mode again.
Param changes clear only the pending layer, so a failed pending render
(spec raise, hull<3, 0 points) can no longer blank the armed display for
the rest of a run.

Re-arm carries the LIVE ownership flag (or local when idle), never the
sticky _last_raster_source, which survives Stop and would hand a freshly
drawn pattern to BLACS.

Dead-zone shading: the image and all overlays sit at z=0 in insertion
order, and ViewBox does not re-bump z above -100 -- at z=-10 the shading
painted behind the opaque camera image and was invisible. Image pinned to
-1, shading to -0.5, and it recomputes when the frame shape changes.

Plus: give_remote_control() (mirror of take_local_control, so the Control
axis is settable from either screen), a third button face 'Give to BLACS',
and tooltips that stop describing the deleted last-stepper-wins world."
```

---

## Task 4: Spec corrections (three adjudicated)

**Files:**
- Modify: `docs/superpowers/specs/2026-08-07-raster-arming-design.md`

- [ ] **Step 1: §5 never-stepped guard — describe what the code now does**

Find the §5 bullet requiring a typed `raster_not_stepped` error (around lines 229-232, "Guard the never-stepped case...") and replace its prescription with: on `_raster_index - 1 < 0`, `raster_point_meta` reports `point_index: -1` (honest "not on a path point yet") with `target_xy` = the laser's actual cached position. A typed error here would sticky-pause the queue on the first shot of every hand-driven run; fabricating point 0's coordinates would be plausible-and-wrong in the h5. Note both rejected alternatives in one sentence each.

- [ ] **Step 2: §9 deploy order — name the real mechanism**

Replace the paragraph claiming "BLACS only sends `raster_current_point` in Local mode, and an old GUI would answer `unknown_connection`" (no such connection exists — §5 was revised to reuse plain `move_to_next`) with the true hazard: **deploying BLACS first, then flipping Control to Local, sends `disarm_raster` to an old GUI that still implements it as `stop_raster()` — destroying the armed path — after which the gated worker never re-arms and every shot sticky-pauses the queue on `raster_not_active`.** Conclusion unchanged: GUI first, then BLACS.

- [ ] **Step 3: §3/§4 — drop-count logging and the owner topic**

In §3, change "the drop lands in the BLACS log through this reply" to name the mechanism: the worker logs `armed`/`dropped` from the arm reply at both arm call sites (Task 5). In §4, add one paragraph: ownership is mirrored tab↔GUI via the `raster_owner` PUB value (`local`/`remote`/`none`); the tab checkbox is driven from it and each real change routes through `update_raster_control`, so widget and worker can never disagree silently. Note the deliberate rule: an incoming `local` unticks the box only while a raster is armed — an idle GUI must not fight the operator's choice of Control=BLACS for position feeding.

- [ ] **Step 4: Commit**

```bash
git commit --only docs/superpowers/specs/2026-08-07-raster-arming-design.md -m "docs(spec): correct never-stepped meta, deploy-order mechanism, owner mirroring

The raster_current_point rationale in section 9 described a connection that
was designed out before implementation; the real ordering hazard is
disarm_raster hitting an old GUI that still destroys the path. Section 5
now describes the honest -1+real-position never-stepped record the code
ships. Section 4 documents the raster_owner PUB value and the mirror rule."
```

---

## Task 5: Worker — release on connect, log the drop, record fire-in-place, precise gate

**Files:**
- Modify: `userlib/user_devices/RasteringDevice/blacs_workers.py` in the worktree (`:11-12`, `:206-238`, two arm sites, `:371-381`)

**Interfaces:**
- Consumes: in-place reply keys `in_place`/`target_xy`/`frame` from Task 2.
- Produces: h5 `/data/<dev>/raster` gains `in_place` attr for fire-in-place shots.

- [ ] **Step 1: Push the release on every sync while local**

In `_sync_raster_mode_to_gui`, replace the disable-branch guard (`:228`):

```python
        # Disabled: disarm only if the GUI might be armed on our behalf. A
        # spinbox jiggle with the box unchecked must not touch the GUI.
        if was_enabled or self._raster_armed:
```

with:

```python
        # Disabled or Control=Local. Under local, push the release
        # UNCONDITIONALLY: connect_to_remote lands here with was_enabled=None
        # and _raster_armed freshly cleared, so the old guard sent nothing --
        # a restored Control=Local never reached the GUI, whose ownership
        # flag then let move_to_next keep advancing (shots_per_step times
        # faster, since local queries every shot). The was_enabled/_raster_armed
        # guard remains for the blacs-control case only (spinbox jiggle with
        # Raster Mode off must not touch the GUI).
        if self.raster_control == "local" or was_enabled or self._raster_armed:
```

- [ ] **Step 2: Log the arm reply's armed/dropped counts (both sites)**

In `_sync_raster_mode_to_gui`, after `self._check_response(response, "raster_arm(settings)")` (`:212`), and in `_advance_raster`, after `self._check_response(response, "raster_arm")`, add (identically, adjusted for the surrounding `try` indent):

```python
                dropped = response.get("dropped") if response else None
                if dropped:
                    # The GUI drops points outside motor travel at arm time;
                    # without this line the drop is visible only on the GUI
                    # status bar, never to an operator watching BLACS.
                    self.logger.info(
                        f"Raster armed with {response.get('armed')} points; "
                        f"{dropped} dropped (outside motor travel).")
```

- [ ] **Step 3: Whitelist `in_place`**

Replace `:11-12`:

```python
RASTER_META_KEYS = ("point_index", "path_len", "target_xy", "frame",
                    "calibration_matrix", "calibration_offset")
```

with:

```python
# in_place: the GUI fired the shot at the laser's current position (operator
# holds control, nothing armed). Without it in the whitelist a fire-in-place
# shot is byte-identical in the h5 to a shot with no raster at all.
RASTER_META_KEYS = ("point_index", "path_len", "target_xy", "frame",
                    "calibration_matrix", "calibration_offset", "in_place")
```

- [ ] **Step 4: Gate on coordinate writes, not table presence**

Replace the gate condition (`:373`):

```python
                if self.raster_control == "local":
```

with:

```python
                coord_writes = any(
                    c == COMPOUND_XY or c in COORD_PAIR for c, _ in writes)
                # Raise only when the sequence actually programs COORDINATES:
                # gating on mere table presence would fire, with a message
                # about coordinates, on a future non-coordinate child.
                if self.raster_control == "local" and coord_writes:
```

- [ ] **Step 5: Syntax-check and commit**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n labscript --no-capture-output python -m py_compile C:/Users/radmo/labscript-suite/.claude/worktrees/raster-control/userlib/user_devices/RasteringDevice/blacs_workers.py`
Expected: exit 0.

```bash
git -C C:/Users/radmo/labscript-suite/.claude/worktrees/raster-control commit --only userlib/user_devices/RasteringDevice/blacs_workers.py -m "fix(RasteringDevice): push Local release on connect; record fire-in-place; log drops

connect_to_remote's re-sync fell into the disable branch with was_enabled
None and _raster_armed freshly cleared, so a restored Control=Local was
never pushed to the GUI -- whose ownership flag then let move_to_next keep
advancing, shots_per_step times faster since local queries every shot. The
release is now unconditional while Control=Local.

in_place joins RASTER_META_KEYS: without it a fire-in-place shot was
byte-identical in the h5 to a shot with no raster at all. The arm reply's
armed/dropped counts now reach the BLACS log at both arm call sites -- the
code comment claimed they did; nothing read the reply. The Control=Local
buffered gate now fires only when the sequence actually programs
coordinates, not on mere table presence."
```

---

## Task 6: Tab — a mirror that syncs the worker, driven by `raster_owner`

The committed mirror repaints the checkbox inside a `blockSignals` sandwich, so the worker's `raster_control` never changes — tab and worker diverge silently, and the divergence re-opens the ownership-seizure hole (stale worker `"blacs"` → reconnect arm → GUI `:484` seizes). It also infers ownership from `raster_mode`, which cannot express a locally-owned continuous raster. Replace it.

**Files:**
- Modify: `userlib/user_devices/RasteringDevice/blacs_tabs.py` in the worktree (`:79`, `:439-454`)

**Interfaces:**
- Consumes: `raster_owner` PUB value from Task 2 (`local`/`remote`/`none`).

- [ ] **Step 1: Subscribe**

Replace `:79`:

```python
STATUS_TOPICS = ["raster_mode", "calibration_status", "raster_progress"]
```

with:

```python
STATUS_TOPICS = ["raster_mode", "calibration_status", "raster_progress",
                 "raster_owner"]
```

- [ ] **Step 2: Replace the lossy mirror**

In `_on_status_received`, replace the `raster_mode` branch's mirror block (`:448-454`):

```python
            # Mirror ownership the operator may have changed at the GUI. The
            # blockSignals sandwich is what keeps a PUB repaint from firing the
            # operator's toggled slot -- same guard restore_save_data uses.
            if value in ("manual", "step", "continuous"):
                self.raster_control_box.blockSignals(True)
                self.raster_control_box.setChecked(value != "manual")
                self.raster_control_box.blockSignals(False)
```

with:

```python
            # Ownership is mirrored from the raster_owner topic, not from
            # here: raster_mode conflates run-mode with ownership (a locally
            # owned CONTINUOUS raster publishes "continuous"), so inferring
            # Control from it re-ticked boxes the operator had just unticked.
            # Remember the run-state for the owner branch's armed check.
            self._last_raster_mode_value = value
```

and add a new branch after the `raster_mode` one:

```python
        elif topic == "raster_owner":
            # Mirror WITHOUT blockSignals: the toggled slot firing IS the
            # worker sync (update_raster_control), and suppressing it is how
            # the widget and self.raster_control diverged -- a stale worker
            # "blacs" re-arms on reconnect and the GUI's already-armed branch
            # seizes the raster back from the operator. setChecked only fires
            # on an actual change and the GUI publishes a steady value, so
            # the loop terminates: tick -> update -> arm/disarm -> same value
            # published -> no further change.
            armed = getattr(self, "_last_raster_mode_value", "idle") in (
                "manual", "step", "continuous")
            if value == "remote" and not self.raster_control_box.isChecked():
                self.raster_control_box.setChecked(True)
            elif (value == "local" and armed
                    and self.raster_control_box.isChecked()):
                # Only while a raster is armed: an idle GUI publishing
                # "local" must not fight the operator's Control=BLACS choice
                # for pattern-less remote position feeding.
                self.raster_control_box.setChecked(False)
            # "none": ownership unset at the GUI -- leave the choice alone.
```

- [ ] **Step 3: Syntax-check and commit**

Run: `& "$env:USERPROFILE\miniconda\condabin\conda.bat" run -n labscript --no-capture-output python -m py_compile C:/Users/radmo/labscript-suite/.claude/worktrees/raster-control/userlib/user_devices/RasteringDevice/blacs_tabs.py`
Expected: exit 0.

```bash
git -C C:/Users/radmo/labscript-suite/.claude/worktrees/raster-control commit --only userlib/user_devices/RasteringDevice/blacs_tabs.py -m "fix(RasteringDevice): Control mirror syncs the worker, driven by raster_owner

The old mirror repainted the checkbox inside blockSignals, so the worker's
raster_control never followed -- tab and worker diverged silently, and a
stale worker 'blacs' re-arms on reconnect, whose already-armed branch
seizes the raster back from the operator: the exact bug class the Control
split exists to kill. It also inferred ownership from raster_mode, which
cannot express a locally-owned continuous raster, re-ticking boxes the
operator had just unticked.

The mirror now follows the dedicated raster_owner value and fires the real
toggled slot (no blockSignals) -- the slot IS the worker sync, and it
terminates because setChecked only fires on change against a steady
published value. An incoming 'local' unticks only while a raster is armed,
so an idle GUI cannot fight Control=BLACS chosen for position feeding."
```

---

## Deploy and verify (operator)

- [ ] Restart the **rastering GUI first**, then BLACS (spec §9 — old-GUI `disarm_raster` destroys paths).
- [ ] Draw a hull with the calibration loaded → **translucent red shading is actually visible** over the left strip (Task 3 was committed with it invisible; do not skip this).
- [ ] Take local control at the GUI mid-queue → tab Control box unticks by itself within ~1 s AND the next sequence with explicit coords **raises** (proves the worker followed, not just the widget).
- [ ] Edit a step-size while armed → grey open circles appear as a separate layer; armed dots never vanish; status line reports `armed N | pending M`.
- [ ] Press Stop, redraw, press Re-arm → the GUI's Control indicator stays **Local** (not BLACS).

## Deliberately not fixed

- `_execute`'s uncalibrated passthrough (Hypothesis B hardening) — out of scope; tracked in `open-items.md`.
- `raster_control: "blacs"|"local"` h5 stamp — still optional; `in_place` + point meta cover provenance.
