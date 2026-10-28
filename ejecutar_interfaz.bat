@echo off
setlocal

:: Asegurar directorio de trabajo en la carpeta de la interfaz
cd /d "%~dp0"

title TMM - Interfaz Grafica

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en "%~dp0venv".
    echo Por favor ejecuta primero "instalar_entorno.bat" para crearlo e instalar las dependencias.
    echo.
    pause
    exit /b 1
)

echo Iniciando interfaz grafica de TMM...
"venv\Scripts\python.exe" interfaz_tmm.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVISO] La aplicacion finalizo con codigo de salida %ERRORLEVEL%.
    pause
)
