@echo off
setlocal enabledelayedexpansion

:: ==============================================================================
:: Script de instalacion de entorno virtual para TMM-interfaz
:: ==============================================================================

title Instalador de Entorno Virtual - TMM Interfaz
color 0F

:: Asegurar que el directorio de trabajo sea la carpeta donde esta este archivo .bat
cd /d "%~dp0"

echo ==============================================================================
echo           CONFIGURACION DEL ENTORNO VIRTUAL PARA TMM-INTERFAZ
echo ==============================================================================
echo Carpeta destino: %~dp0
echo.

:: ------------------------------------------------------------------------------
:: Paso 1: Deteccion del interprete de Python en el sistema
:: ------------------------------------------------------------------------------
echo [1/4] Buscando interprete de Python...
set "PYTHON_CMD="

:: Probar con 'py -3' (Python Launcher de Windows)
py -3 -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py -3"
    goto :python_ok
)

:: Probar con 'py'
py -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py"
    goto :python_ok
)

:: Probar con 'python' directo (verificando que no sea el stub de Windows Store)
python -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
    goto :python_ok
)

:: Probar rutas comunes de instalacion en LocalAppData
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python312\python.exe""
    goto :python_ok
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python311\python.exe""
    goto :python_ok
)
if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python310\python.exe""
    goto :python_ok
)

:: Probar rutas comunes en Program Files
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_CMD="%ProgramFiles%\Python312\python.exe""
    goto :python_ok
)
if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_CMD="%ProgramFiles%\Python311\python.exe""
    goto :python_ok
)

echo [ERROR] No se encontro ninguna instalacion valida de Python.
echo Asegurate de tener Python instalado y accesible en tu sistema.
goto :error_exit

:python_ok
echo [OK] Python detectado: !PYTHON_CMD!
!PYTHON_CMD! --version
echo.

:: ------------------------------------------------------------------------------
:: Paso 2: Creacion del entorno virtual 'venv'
:: ------------------------------------------------------------------------------
echo [2/4] Verificando entorno virtual 'venv'...
if exist "venv\Scripts\python.exe" (
    echo [INFO] El entorno virtual 'venv' ya existe en esta carpeta.
) else (
    echo [INFO] Creando entorno virtual en "%~dp0venv"...
    !PYTHON_CMD! -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Fallo al crear el entorno virtual con: !PYTHON_CMD! -m venv venv
        goto :error_exit
    )
    if not exist "venv\Scripts\python.exe" (
        echo [ERROR] El entorno virtual no se creo correctamente.
        goto :error_exit
    )
    echo [OK] Entorno virtual creado exitosamente.
)
echo.

:: ------------------------------------------------------------------------------
:: Paso 3: Actualizacion de pip e instalacion de librerias de requirements.txt
:: ------------------------------------------------------------------------------
echo [3/4] Actualizando gestor de paquetes pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [OK] Pip actualizado.
echo.

echo [INFO] Instalando librerias desde requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] No se encontro el archivo requirements.txt en "%~dp0"
    goto :error_exit
)

"venv\Scripts\pip.exe" install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Ocurrio un error al instalar los paquetes de requirements.txt
    goto :error_exit
)
echo [OK] Todas las librerias fueron instaladas correctamente.
echo.

:: ------------------------------------------------------------------------------
:: Paso 4: Comprobacion de integridad de las librerias instaladas
:: ------------------------------------------------------------------------------
echo [4/4] Verificando importacion de dependencias...
"venv\Scripts\python.exe" -c "import numpy, scipy, torch, matplotlib, openpyxl; print('[OK] Verificacion exitosa: numpy, scipy, torch, matplotlib y openpyxl importados sin errores.')"
if %ERRORLEVEL% neq 0 (
    echo [ADVERTENCIA] Algunas dependencias podrian no haberse cargado correctamente.
)
echo.

:: ------------------------------------------------------------------------------
:: Finalizacion exitosa
:: ------------------------------------------------------------------------------
echo ==============================================================================
echo [EXITO] El entorno virtual esta completamente listo para usar.
echo Ubicacion del entorno: "%~dp0venv"
echo.
echo Para ejecutar la interfaz grafica puedes:
echo   1. Ejecutar el archivo "ejecutar_interfaz.bat"
echo   2. O desde la consola: venv\Scripts\python.exe interfaz_tmm.py
echo ==============================================================================
echo.
pause
exit /b 0

:error_exit
echo.
echo ==============================================================================
echo [ERROR] La instalacion no se completo. Revisa los mensajes anteriores.
echo ==============================================================================
echo.
pause
exit /b 1
