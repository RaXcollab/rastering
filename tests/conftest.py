"""pytest collection-phase bootstrap.

LOAD-BEARING: this file MUST be ``conftest.py`` in the tests/ root so that
pytest imports it BEFORE any test module. The eager rotpy block below
sets up the Windows DLL load order before numpy / PyQt5 are pulled in by
any test (importing PyQt5 first triggers a Windows DLL-loader deadlock
at rotpy's .pyd resolution -- see camera.py module top + the
``reference_rotpy-pyqt-dll-load-order.md`` memory for the full mechanism).

Symptom this prevents: pytest hangs forever at the first test that does
``from ui import RasterMainWindow`` (or similar) -- the import never
completes because numpy/PyQt5 was already loaded during pytest collection
of an earlier test file (typically ``test_exposure_slider_camera.py``
which imports ``camera_settings_dock`` -> PyQt5 at module top).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Run tests with QT_QPA_PLATFORM=offscreen so PyQt5 doesn't try to open a
# real display under a CI / headless invocation. The
# test_exposure_slider_camera.py file sets this too via ``setdefault``
# (load-order-independent); we set it here for tests that don't.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import rotpy  # noqa: F401  -- side effect: register DLL paths
    from rotpy import system as _rotpy_system  # noqa: F401  -- eager .pyd load
    from rotpy import camera as _rotpy_camera  # noqa: F401  -- eager .pyd load
except (ImportError, OSError):  # pragma: no cover -- production envs always have rotpy
    # ImportError covers .pyd DLL load failures (Python wraps Windows WinError).
    # OSError covers rare cases where ``os.add_dll_directory(p)`` inside rotpy's
    # __init__.py raises -- rotpy guards this with ``isdir(p)`` today but the
    # broader catch is defensive against future changes.
    pass
