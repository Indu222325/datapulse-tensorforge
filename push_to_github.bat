@echo off
rem Data Pulse - send this folder to a new, EMPTY GitHub repository. Double-click this file (Git must be installed).
setlocal
cd /d "%~dp0"
echo First create an empty repository on github.com (do NOT add a README), then copy its address.
echo It looks like:  https://github.com/YOUR_USERNAME/datapulse-tensorforge.git
echo.
set /p REPO=Paste the repository address and press Enter: 
set /p GNAME=Your name: 
set /p GEMAIL=Your email: 
if "%REPO%"=="" goto :fail
if exist .env (
  echo A .env file exists. It is ignored by git, so the key will not be uploaded.
)
git init
git config user.name "%GNAME%"
git config user.email "%GEMAIL%"
git add .
git commit -m "Data Pulse MVP"
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin %REPO%
git push -u origin main
if errorlevel 1 goto :fail
echo.
echo Done. Open the repository page in your browser and check the files are there.
pause
exit /b 0
:fail
echo.
echo Something failed. Copy all the text in this window and send it to Claude.
pause
exit /b 1
