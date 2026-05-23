# Rastering GUI

Laser ablation rastering control: Thorlabs Z912 motors, IDS uEye camera, pattern-based rastering.

## Python Environment

- This GUI uses conda env **`rastering`**, NOT `labscript`:
  `source ~/miniconda/etc/profile.d/conda.sh && conda activate rastering`
- **Tests:** only `pytest tests/test_raster_pathmodel.py` is camera-safe (pure path/controller logic; runs in CI). `test_command_queue.py` and `test_raster_goto_handlers.py` import `ui.py` → open the uEye camera → **HANG when the GUI/camera is busy** — never run them (or the whole `tests/` dir) while the rastering GUI runs. Use `python -m py_compile` for syntax. Tests are standalone-runnable.

## Worktrees

- This is the **live** rastering GUI — a git worktree of `RaXcollab/rastering` on `main`, which the operator runs between shots. Dev happens on **topic-branch worktrees** of the same repo (`git worktree add`); never park half-applied work on `main`.

## BLACS Integration

This GUI is integrated into the BLACS experiment control system (labscript-suite). The ZMQ server in `raster_controller.py:_zmq_loop()` speaks the **v2 RemoteControl protocol** (2026-05-23 cutover).

Module-level `_RasteringV2Server(RemoteControlServerBase)` (imported from parent's `userlib/external_gui_lib/zmq_v2.py`) handles REQ-REP dispatch via `@handler`-decorated methods. PUB-SUB stays raw `zmq.PUB` (topics already match spec §4.1 `{conn}_monitor`). Special cases:

- `move_to_next` iterator-end was non-spec v1 `"FINISHED"`; now SUCCESS + `extra.finished=True` (spec §1.3 fixes 5-token enum). BLACS-side `RasteringWorker.transition_to_buffered` checks `response.get("finished") is True`.
- `arm_raster` returns SUCCESS + `extra.mode={"continuous","step"}`.
- `timeout_sec` lives in v2 `args` dict (Q2). Defaults to 10s.

- **BLACS device code**: `C:\Users\radmo\labscript-suite\userlib\user_devices\RasteringDevice\`
- **Full integration docs**: see `BLACS_Integration_Notes.md` in that directory
- **Canonical v2 protocol spec**: `C:\Users\radmo\labscript-suite\docs\remotecontrol-zmq-protocol-v2.md`
- **DEPRECATED v1 contract**: `C:\Users\radmo\labscript-suite\userlib\user_devices\BLACS_COMMUNICATION_CONTRACT.md` (archaeological only; v2 servers refuse v1 envelopes per Q4 hard sunset)
- **BLACS agent**: `amo-expert` in `C:\Users\radmo\labscript-suite\.claude\agents\`

**If changing ZMQ connection names or PUB-SUB topics**, the BLACS device must also be updated. See the BLACS Integration section in the `ablation-tech` agent prompt for the full list of shared connection names.

## Key Files

- `raster_controller.py` — Central controller: motor commands, raster state machine, ZMQ server
- `ui.py` — PyQt5 GUI (no ZMQ, no hardware direct calls)
- `config.py` — All configuration (ports, hardware serials, camera params)
- `hardware.py` — Thorlabs Kinesis motor interface
- `camera.py` — IDS uEye camera interface
- `raster_paths.py` — Path generation algorithms (grid, spiral, convex hull)
