# Rastering GUI

Laser ablation rastering control: Thorlabs Z912 motors, IDS uEye camera, pattern-based rastering.

## BLACS Integration

This GUI is integrated into the BLACS experiment control system (labscript-suite). The ZMQ server in `raster_controller.py:_zmq_loop()` speaks the RemoteControl JSON protocol.

- **BLACS device code**: `C:\Users\radmo\labscript-suite\userlib\user_devices\RasteringDevice\`
- **Full integration docs**: see `BLACS_Integration_Notes.md` in that directory
- **BLACS communication protocol**: read `C:\Users\radmo\labscript-suite\userlib\user_devices\BLACS_COMMUNICATION_CONTRACT.md`
- **BLACS agent**: `labscript-amo-expert` in `C:\Users\radmo\labscript-suite\.claude\agents\`

**If changing ZMQ connection names or PUB-SUB topics**, the BLACS device must also be updated. See the BLACS Integration section in the `ablation-tech` agent prompt for the full list of shared connection names.

## Key Files

- `raster_controller.py` — Central controller: motor commands, raster state machine, ZMQ server
- `ui.py` — PyQt5 GUI (no ZMQ, no hardware direct calls)
- `config.py` — All configuration (ports, hardware serials, camera params)
- `hardware.py` — Thorlabs Kinesis motor interface
- `camera.py` — IDS uEye camera interface
- `raster_paths.py` — Path generation algorithms (grid, spiral, convex hull)
