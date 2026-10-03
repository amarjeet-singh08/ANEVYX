@echo off
:: Ensure we are in the directory where the batch file is located
cd /d "%~dp0"
color 0B
echo ===================================================
echo               STARTING ANEVYX SYSTEM
echo ===================================================
echo.

echo [0/3] Cleaning up previous sessions...
:: Kill any process using port 8000 (Backend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)
:: Kill any process using port 5173 (Frontend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)

echo [1/3] Starting Python Backend AI Server...
:: Use %~dp0 (current script directory) instead of %cd% which can break if run as Administrator
start "ANEVYX Backend" cmd /k "set PYTHONPATH=%~dp0&& python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo [2/3] Starting React Frontend Server...
cd frontend
start "ANEVYX Frontend" cmd /k "npm run dev"

echo [3/3] Waiting for servers to spin up...
:: ping is more reliable than timeout when run in various environments
ping 127.0.0.1 -n 5 > nul

echo.
echo Launching your browser to http://localhost:5173 ...
start http://localhost:5173

echo.
echo ===================================================
echo System is running! You can minimize these windows.
echo Close the command windows to stop the servers.
echo ===================================================
pause
