@echo off
REM Find Python in ml_fresh environment
setlocal enabledelayedexpansion

set "CONDA_ROOT=C:\Users\AIML\anaconda3"
set "ML_FRESH_PATH=%CONDA_ROOT%\envs\ml_fresh"

echo Checking ml_fresh environment at: %ML_FRESH_PATH%
echo.

REM Check if directory exists
if not exist "%ML_FRESH_PATH%" (
    echo ERROR: ml_fresh environment directory does not exist
    exit /b 1
)

echo Directory structure:
dir /b "%ML_FRESH_PATH%"
echo.

REM Look for python anywhere in the directory
echo Searching for python.exe:
dir /s /b "%ML_FRESH_PATH%\python.exe" 2>nul || echo (Not found in standard locations)
echo.

echo Searching in Library\bin:
if exist "%ML_FRESH_PATH%\Library\bin" (
    dir /b "%ML_FRESH_PATH%\Library\bin\python*"
) else (
    echo Library\bin does not exist
)
echo.

echo Searching in Scripts:
if exist "%ML_FRESH_PATH%\Scripts" (
    dir /b "%ML_FRESH_PATH%\Scripts\python*"
) else (
    echo Scripts does not exist
)

pause
