@echo off
REM Search for the actual python executable
set "CONDA_ROOT=C:\Users\AIML\anaconda3"
set "ML_FRESH=%CONDA_ROOT%\envs\ml_fresh"

echo Searching for python.exe in ml_fresh environment...
echo.

REM Use dir /s to recursively search, but limit depth
for /r "%ML_FRESH%" %%F in (python.exe) do (
    echo Found: %%F
)

echo.
echo Checking specific locations:
if exist "%ML_FRESH%\python.exe" echo ✓ Found at root
if exist "%ML_FRESH%\Scripts\python.exe" echo ✓ Found in Scripts
if exist "%ML_FRESH%\Library\bin\python.exe" echo ✓ Found in Library\bin
if exist "%ML_FRESH%\Library\mingw-w64\bin\python.exe" echo ✓ Found in Library\mingw-w64\bin
if exist "%ML_FRESH%\Lib\venv\scripts\nt\python.exe" echo ✓ Found in Lib\venv (but this is likely broken)

echo.
pause
