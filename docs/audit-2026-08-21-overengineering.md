# Over-engineering audit — 2026-08-21 (ponytail-audit, not yet applied)

Repo-wide scan after a full-source read. Every "no caller" claim below was
grep-verified on 2026-08-21 (excluding `Old Code/`). Findings ranked biggest
cut first. **Nothing here has been applied.**

Scope: complexity only. Correctness/security/perf were out of scope.

## Findings

| # | Tag | Cut | Replacement | Where |
|---|-----|-----|-------------|-------|
| 1 | delete | Entire `Old Code/` — 3,220 lines of superseded legacy (toolbox.py, gui.py, gui2.py, launch_camera_*.py, 2 old .ui, pngs, stale json). Nothing current imports it; git history keeps it | nothing | `Old Code/` |
| 2 | delete | Both executed SDD plans (2,053 lines). Keep the spec (393 ln) + deploy checklist | spec + checklist + code/tests | `docs/superpowers/plans/` |
| 3 | delete | Stale data: 2025–2026 raster/laser logs, position CSVs, `20260707_camera_params.ini` snapshot (`raster_log_enabled=False` writes nothing there now) | nothing | `Logs/`, `20260707_camera_params.ini` |
| 4 | delete | `build_controller()` — hasattr probe + 30-line manual fallback for a factory that always exists | call `create_controller_from_config(config.APP_CONFIG)` directly | `main_rastering.py:14-60` |
| 5 | yagni | `_get()` dual-schema config walker + config.py "Backwards-compatible constants" block (only feeds those fallbacks), ~45 lines | read `cfg.hardware.serial_x` etc. directly | `raster_controller.py:2667-2716`, `config.py:150-162` |
| 6 | delete | Soft-home pathway: `HOME_SOFT_X/Y` enums, `(axis, hard)` dispatch dict, `soft_home()` on Motor/KCube/SimulatedMotor (KCube's just calls hard_home), `home_*_soft` tag entries. No caller passes `hard=False` | one hard-home path, drop the `hard` param | `hardware.py`, `raster_controller.py:141-142,1069-1085,2038-2052` |
| 7 | delete | `JOG_TARGET`: enum, `_execute` branch, `"jog"` tag entries, its test. No `request_*` produces it (UI jogs via `JOG_MOTOR`) | nothing | `raster_controller.py:131,2166-2181`, `tests/test_raster_pathmodel.py:888` |
| 8 | delete | `scripts/build_rotpy.cmd` — builds rotpy against Spinnaker (FLIR); no `rotpy` import anywhere in this uEye repo | nothing | `scripts/build_rotpy.cmd` |
| 9 | delete | One of the near-duplicate remote-arm tests (deploy-checklist open item #3); keep `...arms_as_remote_source` (pins `armed`/`dropped`) | the surviving test | `tests/test_zmq_v2_protocol.py:331,370` |
| 10 | delete | `Motor.lock/unlock/is_locked` — only legacy RasterManager ever called lock(); controller serializes access instead | nothing | `hardware.py:166-172` + 3 guards |
| 11 | delete | Dead fallback creation of `raster_continuous_checkbox`/`raster_step_button` (the .ui provides both) + try/disconnect "runs twice" guards (runs once, in `__init__`), ~22 lines | nothing | `ui.py:844-852,896-906` |
| 12 | delete | `list_device_serials()`, `segments_from_points()` — exported, zero callers | nothing | `hardware.py:141-149`, `raster_paths.py:63-70` |
| 13 | yagni | `emit_rgb` flag threaded config → ui → camera for a mono pipeline nothing flips true, ~10 lines | nothing | `config.py:110`, `camera.py:25,467`, `ui.py` ×3 |
| 14 | delete | `KinesisOptions.verbose` / `hardware.verbose` — threaded through 3 files, never read | nothing | `config.py:45`, `hardware.py:212`, `raster_controller.py:2699` |
| 15 | delete | `MotorCommand.created_ts` — written, never read (seq counter is the tiebreaker) | nothing | `raster_controller.py:178` |
| 16 | delete | `flip_x_checkbox`/`flip_y_checkbox` aliases — assigned, never read; structure test asserts the widgets are gone from the .ui | nothing | `ui.py:444-447` |
| 17 | shrink | `_apply_image_scale(scale=None)` arg-ignoring wrapper + `_img_scale` set-once-never-read | call `_apply_image_mapping()` at the one call site | `ui.py:148,770-772` |
| 18 | shrink | Rotation `k_map`/`k_to_index` literal duplicated 5× | one module constant pair | `camera_settings_dock.py:438,548`, `ui.py:430,669,2340` |
| 19 | shrink | README documents the deleted world (toolbox.py serials, raster_gui2.ui, Pillow, manual scale/offset) | rewrite against current entry points | `README.md` |
| 20 | native | Pillow in README requirements — only `Old Code/gui.py` ever imported PIL | drop the listed dep | `README.md:55` |

**net: ≈ -5,500 lines** (3,220 legacy py + 2,053 executed plans + ~240 live code), **-1 dep**.

## Before applying

- Items 6/7 touch tag whitelists (`home_*_soft`, `"jog"`) — after any cut run
  all four camera-safe suites:
  `pytest tests/test_raster_pathmodel.py tests/test_zmq_v2_protocol.py tests/test_raster_gui_ui_structure.py tests/test_status_strip.py`
- Items 1 and 3 are directory deletes → need explicit operator confirmation
  (lab-wide invariant).
- Item 9: `python -m py_compile ui.py` for anything touching ui.py; never
  import it while the GUI runs.
- Never touch `calibration_data.json`.
- Line refs are 2026-08-21 line numbers; re-grep the named symbols if the
  files have moved since.
