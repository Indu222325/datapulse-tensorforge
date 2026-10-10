@echo off
rem Data Pulse - build the Docker image, start it, test it. Double-click this file (Docker Desktop must be running).
setlocal
cd /d "%~dp0"
set /p API_KEY=Paste your API key and press Enter: 
if "%API_KEY%"=="" goto :fail
echo.
echo [1/4] Building the Docker image. The first build takes several minutes...
docker build -t datapulse .
if errorlevel 1 goto :fail
echo.
echo [2/4] Starting the container on port 8000...
docker rm -f datapulse-local >nul 2>&1
docker run -d --name datapulse-local -p 8000:8000 -e API_KEY=%API_KEY% datapulse
if errorlevel 1 goto :fail
echo.
echo [3/4] Waiting 10 seconds for the service to start...
timeout /t 10 /nobreak >nul
echo.
echo [4/4] Running the automatic checks...
python -m pip install --quiet jsonschema
python scripts\smoke_test.py http://localhost:8000 %API_KEY%
echo.
echo The demo site is open in your browser: http://localhost:8000
start "" http://localhost:8000
echo To stop the container later, run:  docker rm -f datapulse-local
echo.
pause
exit /b 0
:fail
echo.
echo Something failed. Copy all the text in this window and send it to Claude.
pause
exit /b 1
