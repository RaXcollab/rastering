@echo off
REM Helper: build rotpy from source against the installed Spinnaker SDK,
REM into the rastering conda env. Sources vcvars64 to set up the MSVC env.
REM Invoke from any shell (no need for the x64 Native Tools prompt).

setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYEXE=C:\Users\radmo\miniconda\envs\rastering\python.exe"
set "ROTPY_INCLUDE=C:\Program Files\Teledyne\Spinnaker\include"
set "ROTPY_LIB=C:\Program Files\Teledyne\Spinnaker\lib64\vs2015"

if not exist "%VCVARS%" goto :err_vcvars
if not exist "%PYEXE%" goto :err_py
if not exist "%ROTPY_INCLUDE%\Spinnaker.h" goto :err_inc
if not exist "%ROTPY_LIB%\Spinnaker_v140.lib" goto :err_lib

call "%VCVARS%"
if errorlevel 1 goto :err_vcvars_run

echo [build_rotpy] ROTPY_INCLUDE=%ROTPY_INCLUDE%
echo [build_rotpy] ROTPY_LIB=%ROTPY_LIB%
echo [build_rotpy] python=%PYEXE%
echo [build_rotpy] starting source build of rotpy ...

if "%ROTPY_VERSION%"=="" set "ROTPY_VERSION=*"
"%PYEXE%" -m pip install "rotpy==%ROTPY_VERSION%" --no-binary rotpy --no-build-isolation --force-reinstall --no-deps
exit /b %errorlevel%

:err_vcvars
echo [build_rotpy] ERROR: vcvars64.bat not found
exit /b 2
:err_py
echo [build_rotpy] ERROR: rastering env python not found at %PYEXE%
exit /b 3
:err_inc
echo [build_rotpy] ERROR: Spinnaker.h not found under %ROTPY_INCLUDE%
exit /b 4
:err_lib
echo [build_rotpy] ERROR: Spinnaker_v140.lib not found under %ROTPY_LIB%
exit /b 5
:err_vcvars_run
echo [build_rotpy] ERROR: vcvars64 returned non-zero
exit /b 6
