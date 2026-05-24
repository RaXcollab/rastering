# Spinnaker Migration -- Handoff (next rastering code session)

> Snapshot at the close of the IDS uEye -> Teledyne FLIR Spinnaker (rotpy)
> camera migration. Code-complete on `feat/spinnaker-gige`; awaiting
> operator-side validation (V6 calibration re-fit, V8 stress run) and a
> coordinated push of both repos.

## Branch state (read this first)

Two repos, same branch name -- they ship together:

| Repo | Branch | HEAD | Status |
|---|---|---|---|
| Subrepo `GUIs/rastering/` | `feat/spinnaker-gige` | `2a62090` | **16 commits ahead of origin**, unpushed |
| Parent `labscript-suite` | `feat/spinnaker-gige` | `645fa0d` | **1 commit ahead** (parent-doc sync), unpushed |

Push BOTH together when ready:

```bash
cd C:/Users/radmo/labscript-suite/GUIs/rastering && git push -u origin feat/spinnaker-gige
cd C:/Users/radmo/labscript-suite                 && git push -u origin feat/spinnaker-gige
```

## What the next session inherits

- **Code complete.** Steps 1-5 of the migration plan are all merged on
  `feat/spinnaker-gige`. The full plan lives at
  `~/.claude/plans/i-have-gotten-new-wondrous-shore.md` -- read it for
  the design decisions + AUDIT invariants if you need to modify
  `camera.py` / dock / ui.
- **16/16 tests pass** (`python -m pytest tests/` -- under 12 s on this
  rig; includes 2 hardware smoke tests that auto-skip on no-camera).
- **GUI launches cleanly** (`python main_rastering.py`) -- 30+ s with
  zero tracebacks against the live Blackfly S BFS-PGE-16S2M (sn 26120532).
- **`pyueye` is INTENTIONALLY still installed** (operator decision). The
  archived uEye driver `camera_ueye.py` is preserved verbatim as a
  rollback artifact. Do NOT uninstall pyueye without checking with the
  operator first.

## What's NOT yet verified (operator at the rig)

- **V6 calibration re-fit** -- the 2-click click->motor affine is stale
  (different sensor, different optics, different default ROI framing,
  possibly different pixel-axis handedness). Operator must rerun the
  calibration workflow when next at the rig. `calibration_data.json`
  will regenerate.
- **V8 stress run** -- 10+ minutes of full-ROI + full-FPS preview +
  raster with SpinView "Image Incomplete" = 0 watch. Tunes
  `device_link_throughput_limit` / `gige_packet_delay` if needed.
- **Dock interaction sweep** -- every slider / spinbox / combo on the
  redesigned dock against live hardware. Step 4 changed gain from int
  0-100 to float dB, added GigE + Blackfly-Native groups, dropped
  pixel-clock / timing-mode / gain-boost widgets.

## Quick commands

```bash
# Activate the conda env (NOT 'labscript' -- the rastering GUI has its own).
source ~/miniconda/etc/profile.d/conda.sh && conda activate rastering

# Standalone camera smoke -- run this FIRST after any change that touches camera.py.
python GUIs/rastering/scripts/spin_smoke.py        # expect [smoke] SMOKE OK

# Full test suite (16 tests; 2 hardware-gated auto-skip on no-camera).
cd GUIs/rastering && python -m pytest tests/ -v --tb=short

# Launch the GUI.
python main_rastering.py
```

## Critical landmines (read before touching anything)

1. **Windows DLL load order**: rotpy MUST be imported BEFORE numpy /
   PyQt5 in the same process or the Windows DLL loader deadlocks at
   rotpy's `.pyd` resolution. Each entry-point file carries an eager
   `import rotpy; from rotpy import system, camera` block FIRST. Do NOT
   refactor that block into a shared helper (the helper import would
   already be too late). See:
   - `main_rastering.py` module top
   - `ui.py` module top
   - `camera.py` module top
   - `tests/conftest.py` (pytest collection-phase hook -- without it,
     `pytest tests/` hangs indefinitely at the first ui-import test)
   - `~/.claude/projects/c--Users-radmo-labscript-suite/memory/reference_rotpy-pyqt-dll-load-order.md`

2. **AUDIT invariants on camera.py** (numbered in the plan + cited in
   commit messages):
   - **B1**: `SpinCamera.grab()` is bounded by `get_next_image(timeout=
     0.5s)`; only the rotpy timeout code `-2007` converts to None.
     Other `SpinnakerAPIException.spin_error_code` values RE-RAISE so
     real disconnects surface as `status('closed')`.
   - **B2**: `SpinCamera.open()` is exception-safe (`_teardown_partial`
     unwinds on any raise); `close()` idempotent across partial init.
     `_acquiring=True` is set ONLY after `begin_acquisition()` returns.
   - **S1**: `grab()` returns an OWNED ndarray (`frame.base is None`)
     -- asserted under `__debug__` on every frame.
   - **S2**/**S3**: `CameraThread._pending` snapshot+clear is atomic
     under `_params_lock`; APPLY happens OUTSIDE the lock. Every
     `set_*` / `request_*` slot body is solely the mutex-guarded write
     -- slots NEVER touch `self._cam` from the GUI thread.
   - **N2**: Mono8 -> uint8 2-D ndarray (rastering display path
     expects this; non-Mono8 formats need range-scaling before
     `setImage(autoLevels=False)`).
   - **parity-1**: TLStream `StreamBufferHandlingMode=NewestOnly`
     attempted pre- AND post-`begin_acquisition` (firmware-dependent).

3. **Dock spinbox commits**: every spinbox that commits live to the
   camera (fps, exposure, gain, gamma, packet_size, throughput,
   black_level) uses `setKeyboardTracking(False)` so typing "9000"
   emits ONE `valueChanged` (on Enter/focus-out), not 5. Do NOT remove
   this -- spam reaches the camera as mid-typing partial values.

4. **Legacy ini behavior**: `config.APP_CONFIG.camera.camera_params_ini`
   defaults to `camera_params_spin.ini` (Spinnaker schema). The
   legacy `camera_params.ini` (uEye Cockpit) is preserved untouched as a
   rollback artifact. On first GUI launch with no spin .ini, ui.py logs
   a one-shot warning ("legacy 'camera_params.ini' is present but
   '...spin.ini' is not -- legacy file NO LONGER auto-loaded"). The
   operator either:
   - Uses the dock's **Save Config** button -> writes a fresh Spinnaker
     `.ini` at the configured path, OR
   - Uses the dock's **Load Config** to apply the legacy `.ini` manually
     (the absorber auto-migrates legacy keys with deprecation warnings;
     `pixel_clock` / `master_gain` / `enable_gain_boost` / `use_freeze`
     are dropped silently per Parity 2/4/5; `target_fps` is renamed to
     `acq_frame_rate`).

5. **Retired alias slots**: `set_master_gain` / `set_pixel_clock` /
   `set_gain_boost` / `set_prioritize_exposure` were REMOVED from
   `CameraThread` after the Step-4 dock rewrite. A repo-wide grep
   confirmed zero callers. If you re-introduce a caller, the test
   `test_one_camera_call_per_gesture_no_double_fire` will fail (it
   asserts `set_master_gain MUST NOT appear in fake.calls`).

## Files to read first (in order)

1. `~/.claude/plans/i-have-gotten-new-wondrous-shore.md` -- the master
   migration plan with AUDIT invariants + Parity table.
2. `docs/ROTPY_BUILD.md` -- rotpy install / DLL discovery / troubleshooting.
3. `docs/ROTPY_API.md` -- generated rotpy 0.2.1 API surface (use for
   "does Spinnaker have an X node?" lookups -- it's runtime-introspected
   so the call signatures + first-doc-lines are accurate).
4. `camera.py` module-top comment block -- AUDIT invariants + DLL
   load-order rationale.
5. `~/.claude/projects/c--Users-radmo-labscript-suite/memory/` -- look
   for `reference_rotpy-pyqt-dll-load-order.md`,
   `reference_omp-duplicate-runtime-kmp-workaround.md`,
   `reference_cython-pyd-runtime-introspection.md` -- the three
   migration-era memories that document the trapped-and-fixed pitfalls.

## Useful diagnostics

- `python GUIs/rastering/scripts/spin_smoke.py` -- if this FAILS, the
  problem is environmental (rotpy install, DLL paths, GigE link)
  before it's a code problem.
- If `pytest tests/` hangs at "collecting ..." -- the rotpy bootstrap
  in `tests/conftest.py` was edited; check that the rotpy block runs
  BEFORE numpy/PyQt5.
- If `main_rastering.py` paints nothing and never exits -- DLL deadlock.
  Likely a new module was added to the import chain with a `numpy` or
  `PyQt5` top-level import that runs BEFORE camera. Add the eager rotpy
  bootstrap to that module's top.
- If a Blackfly-Native widget (BlackLevel / DefectCorrection) is greyed
  out -- the node is unavailable on this Blackfly model. Not a bug --
  the dock self-disables via the `bl_max > bl_min` check in
  `update_from_camera_info`.

## Open follow-ups (not blockers)

- **NIT 13** (reviewer): `pixfmt_combo.currentIndexChanged` fires the
  pixel-format slot on every arrow-key tick; rapid up/down spam would
  trigger 100 ms-spaced acquisition stop/start cycles. No fix shipped
  -- arrow-key spam is deliberate user action and the run loop's
  `_pending` coalescing limits the actual damage.
- **Step 3 future**: `config.APP_CONFIG.camera.camera_params_ini`
  pointer could be re-pointed at `camera_params.ini` once the operator
  has migrated their settings. Or the absorber could be extended to
  auto-copy a legacy .ini to spin.ini on first run. Neither shipped.

## Commit history (most recent at top)

Subrepo `feat/spinnaker-gige`:

```
2a62090 review fixes: TOP-1/2/3 + SHOULD-FIX A/B/C (Step 5 follow-up)
2aac7f2 tests + docs + requirements.txt: Step 5 -- final migration plumbing
e041a6e camera_settings_dock: Step 4 -- Spinnaker dock redesign
6b746b5 ui+config+camera: Step 3 -- ui.py & config.py Spinnaker redesign
ff80f22 camera: replace __getattr__ legacy alias with @property descriptors
94424b2 camera: review fixes (B-1, B-2, B-3, TOP-1, S-1..S-3, N-1..N-5)
ed5c85e camera, main: V4 unblock -- legacy read-compat + rotpy load-order at main
d713f4b camera: CameraThread.run loop + error throttle (Step 2.5)
4ac3427 camera: SpinCamera real I/O via rotpy (Step 2.3)
79abc98 camera: smoke fix + ROTPY_API.md (post-first-hardware-run)
32e9c7b camera: ini load/save/apply + DEFAULT_INI (Step 2.6)
5506cd9 camera: CameraThread signals/slots (Step 2.4)
665109b camera: CameraConfig schema + legacy-kwargs absorber (Step 2.2)
8b9f332 camera: skeleton + KMP/lazy-rotpy/aliases (Step 2.1)
9d3e91c rotpy: switch to wheel install + add OMP workaround
4ec8e9c data: recover live calibration_data.json (2026-05-20 auto-save)
```

Parent `feat/spinnaker-gige`:

```
645fa0d docs: rastering uEye -> Spinnaker migration update
```
